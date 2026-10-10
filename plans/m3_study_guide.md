# M3 global fine-split 优化详解(学习指南)

commit: 9dc6d488(分支 m3-wip);改动 5 文件,+295/-93。
目标:coarse 时延与 R 线性(fine-split 把每行拆到 24 对核并行),并让跨核 topk 折叠(fold)可用。

## 一、改动文件与职责
| 文件 | 职责 |
|---|---|
| `csrc/attention/indexer_coarse_screen/op_kernel/arch22/indexer_coarse_screen_kernel.h` | SplitCore 分核 / 核间握手 / 窗口调度 |
| `.../indexer_coarse_screen_service_vector.h` | AIC 收分后全部向量工作:主 pass 打分、partial 槽写、fold 折叠、窗口 dedup |
| `.../indexer_coarse_screen_vector.h` | 排序原语:SortAll/MergeSort/**新增 FoldMerge4** |
| `op_host/indexer_coarse_screen_tiling.cpp` | workspace 布局(partialTopk 槽区 48 line) |

## 二、七个根因修复(按依赖顺序)
### 1. SplitCore 零预算核(M3R-A,KH ~305)
blocks<核数时 minBlockPerCore=0 → s2End=s2Idx-1 无符号下溢 → s2LoopEnd=0xFFFFFFFF 野读 507015。
修:零预算核 isEmptyRange=true 干净 return。**教训:uint 下溢在 NPU 上表现为野读 fault,不是除零。**
### 2. SplitCore 平方记账(M3R-B,KH ~340)
else 分支 lastGS1RemainBlockCnt += s2RemainBaseNum(余量)应 += 1(逐 unit 记账);粒度化后方暴露。
### 3. totalBlockNum 粒度(M3U,host/tiling)
over-2K 曾恒 1 块/行而执行侧按 s2BaseNum 跑 → usedCoreNum 被改写、末核尾零调整跨行全扫 3× 冗余。
### 4. poll 事件信用(M3W,SV PollPartialMagic)
WaitFlag(EVENTID) 是**单信用信号量**,spin 循环里每轮消费一次 → 第二轮永等挂死。修:PIPE_ALL 屏障(无信用语义)作 UB 复用护栏。**教训:事件对只能一次 Set/一次 Wait 配平,自旋轮询必须用 PIPE_ALL。**
### 5. fold 炸机真因(MrgSort 段长越界)
官方文档铁证:elementLengths 是 uint16_t[4]、**单位 8B(一个 score/index 对)、取值 [0,4095]**。
旧体传 8192 对作单段(段长越界 + src 读双倍 + tmp 写 24576 floats 超容量)→ VEC param invalid(0x1000)。"时序脆弱"是十轮误诊,真因是 API 参数非法。
修:**FoldMerge4**(vector.h)—— acc 两半 + 槽两半 = 4×2048 单次 MrgSort,全部段长合法;tmp 需 16384 floats(实测 tmpUb_=17408 ✓)。
**教训:API 约束必须查官方文档(asc.gitcode.com MrgSort 页),勿凭记忆。**
### 6. 槽撕裂链(三个叠加修复)
- **M3F7(真根因)**:fold 内 InitSortOutBuf(V 预填 -1)与紧随的 MTE2 槽载入**同址写 acc,缺 V_MTE2 事件** → MTE2 覆写未完成的 -1 预填 → 部分 -1 行逐 launch 漂移。一行 SetWaitFlag<V_MTE2> 全链归零。
- 写点:F2E/F5 —— MTE3 槽写前 PIPE_ALL(排 V/MTE2 产物),写后 MTE3_V+MTE3_MTE2(写排空才许下一 tile 覆写 acc)。
- **M3F4**:Flush 入口 SyncAll —— 消灭 magic 陈旧竞态(poller 在慢核清零前读到上一 launch 残留 magic)。
判决方法论:fold 零槽读=完美稳定 / own 槽读=抖动 → 定界到读侧;三连发 det 探针排序修复效果。
### 7. 窗口配对(M3F3 + M3F9 + M3F10)
- SyncAll **不保证跨核 GM 可见性**(16 卡生产垃圾输出同签名)→ "行读者=行写者同核"是铁律。
- M3F3:每 AIV 私有位图 winRowMask_(主 pass 完整行 + fold 分裂行置位),dedup 只处理本核写过的行。
- M3F9:空域行(upper==0)由 dedup 的 validC==0 恒等分支冗余覆盖(全部偶 AIV 同内容竞态写无害);删除 EmitIdentityRows 的 wBegin>0 启发式(与 SplitCore 空 s2 域分支冲突)。
- M3F10:位图 4→128 字(4096 行):R=512 时 wr>>5=15 越界写坏相邻成员 → 507015。**教训:按索引寻址的每核数组,容量必须 ≥ 最大 R 并加写保护。**

## 三、跨核折叠协议(fold,可复用)
1. 每 pair 双槽 partial 槽(GM workspace,行槽 rowCap×8192F + 16F magic 尾);写点 tile 尾直写(seq=fragIsHead_?1:0)。
2. Kernel 启动 ZeroMyMagics(防 N-1 残留);Flush 入口 PublishMyMagic(同核 MTE3 FIFO 保数据先落地)。
3. 行尾核 defer 记录(isS2End&&fragIsHead_),cstart 反向步进定位行首核(unitStart 单调)。
4. Flush:poll 各贡献核 magic(8M spin 上限 fail-open)→ own 槽直载 acc → 逐槽 FoldMerge4 → Extract+CopyOut。
5. 顺序保障:Flush 入口 SyncAll(全员主 pass 完成)→ 窗口入口三件套(PIPE_ALL+MTE3_MTE2+SyncAll)→ dedup。

## 四、验证链(全绿,可复跑)
- probe_m1a.py 7 案 ALL-PASS(det=True n_diff=0);probe_m3det.py 三连发 det;probe_m3ws.py has_window=2 dump 通道(窗口期 printf 死,勿用);trace_cases.py 12 案逐 case。
- precision_gate_full.py(GATE PASS);test.py --probe 四相位 EXIT=0。
- bench_m3r.py:graph-replay 口径 R∈{1..48}:R≥4 斜率 12.0-14.9 µs/R,R48=741µs,R1-2≈120µs 固定开销平台。

## 五、后续优化杠杆(按证据)
1. **cube 两级 GEMM 融合归约(fp32 精确)** —— fold(压缩后再计算)死刑的正解;不动模型可见输入。
2. **s2BaseSize 增大二期** —— L 主导(5.2µs/千 token),cube 带宽是地板(80-100µs);V 侧缓冲(reduceOutBuf_/brcBuf_)腾出空间。
3. **refine** —— 数值已近完美上限,纯性能:对标 LI 逐管道(当前 4×:fixpipe 73vs272 / aiv_vec 84vs347 / wait_id0 84vs361)。
4. **AI-CPU 开销** —— if-else/标量链在热路径的收敛;host tiling 分支;无需的 GM 拷贝路径(须无损+全绿验证)。
5. **固定开销 ~120µs 平台** —— SyncAll×2/窗口常量段/poll 超时上限的摊薄。

## 六、避坑速查(今晚新增)
- `ninja ... || true` 吞编译错误 → **SYNCED 时间戳停在旧值 = 静默失败**,每轮必核验;
- 函数改名:定义签名与声明必须同步(漏改=构建连环假绿);
- 窗口期(pipe->Reset 后)printf 静默失效 → 用 has_window=2 dump 通道;
- probe_m1a/test.py 只比 mode0 vs mode1(无 torch 真值),"PASS"可能空洞;真值看 precision_gate_full 的召回门 + 整网 AIME;
- ais_bench 正典调用在 .218(aime_go.js,名字模式;绝对路径必 UTILS-MATCH-001)。
