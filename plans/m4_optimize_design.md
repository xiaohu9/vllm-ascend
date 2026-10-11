# M4 优化设计(wait_id0 头号杠杆 + 队列)

## 背景勘测(2026-10-10,代码实证)
- 我方 cube 缓冲常量(KEY_BUF_NUM=3/QUERY_BUF_NUM=2/L0_BUF_NUM=2)与 LI **逐位一致**;
- LI 握手结构(2 预置信用,AIC Wait(V1C1)→mm→Set(C1V1,PIPE_FIX);AIV Wait(C1V1)→ProcessVec→Set(V1C1,PIPE_MTE2))与我院**逐位一致**;
- LI Fixp 与我院**逐字节同构**(GM 直写);
- 结论:LI 对标的均匀 4×(fixpipe 73vs272/aiv_vec 84vs347/wait 84vs361)= **任务固有尺寸差:我方每请求 g=4 组行(LI 单行)**,非结构性劣化;两级 GEMM 融合降级(收益存疑)。

## 改动一:V1C1 信用早还(wait_id0 修复,预期消 wait 361→~100µs)
病机:AIV 在 ProcessVec **全部完成**(scale/reduce/sort/槽写)后才归还 V1C1 信用,AIC(快)被 AIV 排序拖停;而 mm1Res 双缓冲(loop%2)本就允许 AIC 前瞻 2 块。
over-2K decode 下每块恰 1 次 mm1 CopyIn(outerG=CeilDiv(4,16)=1,innerS1=1)→ 载入完即可还信用。

改法(两阶段拆分):
1. KH ProcessBaseBlock AIV 分支:
   `WaitFlag(syncC1V1); vectorService.ProcessVecLoad(info); CrossCoreSetFlag<MODE2, PIPE_MTE2>(syncV1C1); vectorService.ProcessVecCompute(info);`
2. SV:ProcessVec 拆 ProcessVecLoad(偏移计算+init 分支+innerS1 循环的 mm1 CopyIn 与 MTE2_V 等待)+ ProcessVecCompute(DoScale/DoReduce/SortAll/pair-merge/槽写/CopyOut);
   中间状态进成员:cuS1BeginIdxPerAiv_/cuS1ProcNumPerAiv_/mmGmOffset_/weightGmOffset_/outerG_/cuRealAcSeq_/cuBaseS1Idx_/cuBaseS2Idx_(~8 个 uint32/int32)。
3. **死锁防护**:cuRealAcSeq<=0 或 cuS2Len<=0 的空载块,ProcessVecLoad 内必须无条件走到归还点(路径:loop 内 if 跳过也要计数;或 Load 末尾对未载块直接返回)。
4. 正确性界:AIC 最多前瞻 2 块=双缓冲容量;AIC 覆写 buffer b 前提 = 本核已发 V1C1 信用 = Load 已把该 buffer 数据 CopyIn 完毕且 MTE2 排空(PIPE_MTE2 旗语义保证)。
5. 验证:probe_m1a 7/7 → test.py --probe → bench_m3r(R 斜率不变量)→ AIME+哨兵。预期 ≥64K 再降 ~200µs 级。

## 改动二:refine 对标(次优先)
refine 88-114µs 几乎不随 L 变;同法勘测其 AIC/AIV 停等与缓冲深度,LI refine 对标。

## 改动三:小 L 固定开销平台(~115µs vs 旧 40µs)
构成:双 SyncAll(Flush 入口 M3F4 + 窗口入口)+ fold 协议 + poll 上限。
方向:R=1 或全行完整(无 defer)时 Flush 直接退皮(无 poll 无 SyncAll);窗口入口 SyncAll 在无分裂行时按需跳过——需要跨核先行判定(轻量标志位,可在 main pass 统计后经 GM+magic 传递)。
风险:参与集变化影响 SyncAll 语义,须谨慎验证(全零批路径回归)。

## 改动四:AI-CPU/if-else/拷贝清理(随做)
热路径标量链审计(ProcessVec 头部每块 ~20 次标量计算可下沉 tiling);CleanInvalidOutput 等低频路径不修。

## 纪律
每项:改动前单点 bench 基线 → 改动 → 单点对照 → probe_m1a 7/7 → test.py --probe → AIME+哨兵 → 单 commit;无提升即回退。
## M4-1 实施判决(2026-10-10 11:30,已回退)
V1C1 信用早还(ProcessVec 拆 Load/Compute,KH 中间还信用)已完整实施并验证:
- 正确性:probe_m1a 7/7 det=True;test.py --probe 语义 CASES 全 PASS,矩阵失败项与基线逐项同型(均为既有 hw0 NONDET,20-23 项随机波动,见下节);
- 性能:背靠背 A/B(同负载窗口,bench_m3r L64K)R1/R2/R4/R8 逐点零差异(Δ≤2µs 噪声);
- **判决:零收益,已回退**。wait_id0 握手在 mm1Res 双缓冲(loop%2)2 块前瞻下已被 AIC 侧藏住,稳态瓶颈 = AIV 排序段本身(慢段),握手非杠杆。标量 86% 等事件是慢段等待的表现不是原因。
- 产物:补丁 /tmp/{sv,kh}_m4a_v2.h(.m3b/patch_m4a.py + patch_m4a_fix.py 可复现)。

## 既有 NONDET 定性(与 M4-1 无关,gate2ops_v2.log 07:38 已存在)
test.py --probe 矩阵 hw0(has_window=0)配置 R64/R512 出现 run-to-run NONDET(C 差几十元素,refine 放大 10^5-10^6)。生产/bench/AIME 全走 has_window=1(probe_m1a det=True 逐位确定),非生产路径。M3 收口时同矩阵已 22 项 FAIL,非今日回归。**队列:hw0 窗口路径竞态排查**(独立工作项,不阻塞主线)。

## M4 消融/profile 结论(2026-10-10 12:10,R4/256K,占机指示值但横向一致)

msprof 逐管道(op_summary,10 replay 聚合):**所有管道利用率 ≤8.5%**(AIC
scalar 8.5/fixpipe 7.1/mac 2.2;AIV vec 占 busy 36%)——kernel 是串行化/延迟
瓶颈,不是吞吐瓶颈。旧报告的 fixpipe 63%/AIV 排序 80% 是 M3 前结构,已失效。

相位分解(bench_phase.py):窗口相位免费(win=1 vs 0:319.7 vs 317.5);
墙钟 = 固定 ~105µs + 线性 0.83-0.86µs/K;R4→R8 超线性(319.7→542.3,55.6µs/R)。

消融(错结果无妨,只量墙钟):
| 消融 | R4/256K | 归属 |
|---|---|---|
| 完整基线 | 319.7 | — |
| #1 砍 pair-merge/归并 V 段 | 258.3 | **归并 = 61µs(19%)** |
| #2 +#1 再 stub AIC ComputeMm1 | 252.9 | **AIC 侧(mm/fixpipe/key 载入)≈ 5µs,彻底排除** |
| #3 SortAll 后 continue | (device abort,脚本缺陷;未取数) | — |

**判决**:
1. 线性段 220µs ≈ 85 units/pair × ~2.6µs/unit 的串行链,无单一大头;
2. wait_id0 非杠杆(M4-1 背靠背 A/B 零差异,握手被 2 块前瞻藏住);
3. 剪枝类(运行阈值)在 bench 随机数据上不触发(iid 下 P(块全低于 kth)≈e⁻⁸),
   只对生产衰减分布有效,不满足 bench 验收口径;
4. **÷2 必须结构性攻击**:① 4/8-chunk 批量归并(归并段 ÷2-3,上限 ~30µs);
② 深流水(mm1Res 2→4 缓冲 + 种子信用 2→4);③ 固定 105µs 拆解(Flush/fold/
M1a/CopyOut 逐段消融,消融脚本需保持 CleanInvalidOutput 路径可达——#3 教训)。
预期组合上限:线性 220→~110(①+②) + 固定 105→~70 ≈ **326→~180µs**,接近 ÷2;
进一步需 R4→R8 超线性(55.6µs/R)的 workspace/带宽竞争排查。

消融产物:ablate1/2/3.py(可复现);当前树已恢复 9dc6d488 干净基线(SYNCED 12:06)。
## M4-2 实施级设计:4-chunk 批量归并(2026-10-10 12:15)

### 已核事实(全部 grep/sed 实证)
1. SortAll(indexer_coarse_screen_vector.h:189)= Sort32 + MrgSort 归并,输出
   **交错 (value,index) 8B 对**,降序;chunk 排序后 reduceOutBuff[0..1024) 即
   512 对可直接喂 MrgSort4(srcList 按段长单位 8B)。
2. cuS2LenVecAlign ≡ s2BaseSize_=512(cuS2Len∈(0,512] 恒 512;-inf 预填补尾)——
   批内每段恒 512 对,MrgSort4 lengths 恒 [512,512,512,512]。
3. 暂存区 [2V,4V)(reduceOutBuff[s2BaseSize_*2] 起,4096 floats)= 4 槽 × 1024
   恰好容纳;现 pending 单槽同区。事件对:V_MTE2(2)/MTE2_V(2) 护每次 MTE2 拷贝。
4. MergeSort(acc, virTopK, src, srcLen, tmp) 现调用 srcLen=s2BaseSize_*2(2 段
   512 对);批量后 = stageCnt*512 对 ≤ 2048 对 ≤ 3072(2-list 分支安全,永不进
   3-segment;M3 v11 教训红线)。
5. MrgSort4 段长 [0,4095] 合法;fragEnd 残批 stageCnt<4 用 0 长度段补齐
   (src 指针指槽 0,len=0 不读)。
6. acc 语义 = 运行 top-4096(降序),批量归并与逐对归并的集合恒等;仅平局序
   可能与现实现不同(等价质量,precision gate/recall 门+AIME 裁决,probe 两模式
   走同一新代码,eq_C 语义不变)。
7. 成员:pairFilled_[8] → uint32_t stageCnt_[8] 同位置重置点(两 init 分支)。

### 改动点(service_vector.h 单文件)
- 成员声明:bool pairFilled_[8] → uint32_t stageCnt_[8](保留变量名易查)。
- init 两分支:pairFilled_[i]=false → stageCnt_[i]=0。
- 主循环 pair-merge v2 段整体替换:
  stage:事件对(2)护 DataCopy 到槽 [2V + stageCnt*2*512);++stageCnt;
  if (stageCnt==4 \|\| fragEnd && stageCnt>0):MrgSort4(4×512→tmpSortBuf,
  0 长度补齐)+ MergeSort(acc,4096,tmpSortBuf,stageCnt*512*2?,tmpUb_);stageCnt=0。
  (MergeSort srcLen 单位以现调用 s2BaseSize_*2=1024 对应 2 段 → 单位=对×2,
  即 stageCnt*1024;实施时对照 MergeSort 实现一次确认)
- fragEnd && pairFilled_ 旧兜底段删除(并入上面 rowEnd 分支)。
- 后续 slot-write/defer 记账段不动(acc 内容语义未变)。

### 验证链
probe_m1a 7/7 det → test.py --probe 语义 CASES 全 PASS → bench_phase 背靠背 A/B
(预期归并段 61→~20-30µs,R4/256K 320→~290;L 扫描斜率 0.86→~0.75)→ 干净卡
复测 + AIME 随门。无提升即回退。

### 后续队列(M4-3)
深流水 mm1Res 2→4 缓冲(种子信用 2→4,loop%2→%4);固定 105µs 逐段消融
(Flush/fold/M1a/CopyOut;消融脚本保持 CleanInvalidOutput 可达——ablate3 教训)。
## M4-2 判决(2026-10-10 12:20,已回退)
4-chunk 批量归并:实施后 507015 aicore exception。根因 = MrgSort 单 repeat 队列
上限 1024 对(refine v11 实证红线):4×512=2048 对超限;拆两次 ≤1024 对归并后每
chunk 元素操作 3072 = 与现行 pair-merge v2 恒等(v2 已是 1024 对约束下最优 2-way
结构),且批量方案暂存拷贝翻倍。**判决:无算法收益空间,放弃,已回退。**
暂存区结论留存:tmpUb_[8704..12800)(outerG≡1 pingpong-1 区)与 MergeSort scratch
[0,6144)/v11 丢尾 [4096,8192)、DoScale [0,8704) 零碰撞,可复用。

## M4-3(下一个):AIV 跨块软件流水
机制:现行 AIV 顺序 Load(N)[MTE2 copy]→Compute(N)[sort+merge],MTE2 延迟+拷贝
与 V 归并不重叠。改法:prologue 先发 copy(N0);主循环 {发 copy(N+1)(事件对护,
双缓冲 loop%2 目标);sort+merge(N)}。预期藏 ~0.5-1µs/unit → 线性段 -20-30%。
注意:M4-1 的 Load/Compute 拆分是同块内两段调用,无跨块重叠,故零收益;M4-3 是
真重叠。与 M4-1 拆分正交,可叠加实施。

## M4-3 实施级设计:AIV 跨块软件流水(2026-10-10 下午)

### 机制
现行 AIV 每块:Wait(C1V1)→CopyIn(MTE2,~1µs)→sort+merge(V,~3µs) 全串行。
改 shift-by-one:prologue 发 copy(0);主循环第 N 步 {Wait(C1V1_N);发 copy(N)
进 dbTmpUb[N%2];Wait(MTE2_V,(N-1)%2);consume(N-1);Set(V1C1)};尾声 consume
(last)。MTE2 fetch 与 V compute 跨块重叠,预期藏 ~1µs/unit → 线性段(220µs
@256K)-20~30%。

### 信用链分析(关键,防死锁)
M4-1 教训补充:早还信用下 AIC mm(k) 等 V1C1#k(=AIV Load(k) 后),而 Load(k)
等 C1V1(k)=AIC mm(k) 后 —— 靠 2 个种子信用破环(AIC 先行 mm(0),mm(1)),
实际 lookahead≈1,这就是 M4-1 零收益的深层原因。
M4-3 下 V1C1 在 consume(N-1) 后归还(比 M4-1 更晚),AIC lookahead 仍≥1;
C1V1_N 阻塞只影响该单元的 overlap 生效与否,不引入死锁(计数旗,Wait/Set
次数不变:AIV 每 block 仍 1 Wait + 1 Set,仅时点平移)。
GM 覆写安全:copy N 用 buffer N%2;consume(N-1) 数据已在 UB;V1C1 归还后
AIC 才可 mm(N+1) 覆写 (N+1)%2 —— 与 N、N-1 的 UB 副本无交涉。

### 实施(M4-1 拆分为地基,存 /tmp/{sv,kh}_m4a_v2.h)
1. 恢复 M4-1 的 ProcessVecLoad/Compute 拆分(KH 中间还信用改为尾还,见下)。
2. KH 主循环改 shift-by-one:循环内 AIV 分支 =
   {Wait(C1V1); ProcessVecCopyIn(N); ProcessVecConsume(N-1); Set(V1C1)};
   循环前 prologue(Wait+C copyIn(0)),循环后 epilogue(consume(last)+Set)。
   需要 KH 拿到块总数(SplitCore 后本核 blockNum)与首块/末块特判。
3. ProcessVec 拆三段:CopyIn(纯 MTE2 发起,含 over-2K TMPUB wait)/Consume
   (DoScale 起)/跨调用状态成员(reduceOutBuff 段 + cuS2LenVecAlign + isS2End
   + init 分支标志,详见 M4-1 补丁的 8 成员清单)。
4. init 分支(InitSortOutBuf 等)放 prologue/首块 consume 前。
5. dbTmpUb pingpong 事件序:copy N 用 pingpong=N%2,consume N-1 用 (N-1)%2,
   事件 id 0/1 语义不变。

### 验证链
probe_m1a 7/7 det → test.py --probe 语义 CASES → bench_phase 背靠背(看 L 扫描
斜率 0.86→目标 ≤0.70)→ 干净卡复测 + AIME 随门。无提升即回退。

### 迁移备注(2026-10-10)
.82/.62/.162 下周一退役。已迁:.206(/data 权重 378G rsync 中+repo+小文件)、
.202(/home 小文件,权重不挤占 /data,周一商量)。镜像 tar 8.7G 本地上传两机。
新机 16 卡 910B,驱动/CANN 栈在;.206 sshd AllowTcpForwarding 已修(vscode 远程)。
容器安装:container.sh 在迁移完成的 /home/l00955958,镜像 load 后按 .82 同款
容器名/挂载起。M4-3 收官验证应在 .206 新容器跑(或 .82 退役前)。
## M4-3 判决(2026-10-10 13:20,已回退保绿)
实施完成+编译过,首跑 507014 死锁。根因(分析闭环):over-2K 时 Load 每块
WaitFlag(TMPUB),TMPUB 由 consume 尾部 SetFlag;M4-3 把 consume(k) 推迟到迭代
k+1,Load(k+1) 在同迭代先于 Consume(k) 等它末尾的 TMPUB = 自等自死锁。
更深结论(数学性):MergeSort scratch 固定 [0,6144) ⊂ region-0 拷贝目标 [0,8704),
任何 region-0 拷贝与归并重叠都冲突 → **full 流水线需先迁 MergeSort scratch
(扩 outQueue_ +20KB,须核 UB 总预算)或退半流水(仅 region-1 拷贝重叠,~0.5µs/unit)**。
补丁链:patch_m4c.py(m4c sub() 曾漏 .replace 已修)+ patch_m4c_fix/fix2;当前树已回退 9dc6d488。

## M4 剩余可行路径(按确定性排序)
1. **扩 outQueue_ 迁 MergeSort scratch** → 解锁 M4-3 full 流水(预期线性段 -20~30%);
2. 固定 105µs 逐段消融(Flush/fold/M1a/CopyOut);
3. R4→R8 超线性(55.6µs/R)workspace 竞争排查;
4. hw0 NONDET 竞态排查(非生产路径)。
每项纪律不变:实施→probe 7/7→语义 CASES→背靠背 A/B→干净卡+AIME 随门→达标才 commit。
## 二轮外审 + 收敛(2026-10-10 15:50)
外审两轮结论:M4-3 名义流水实为串行(事件信用单 ID Set/Wait 严格配对,不支持跨块流水),
forceLast 未生效致末块丢失(P0)—— M4-3 方向正式判死。树已回干净基线 + 三项独立修复
(6.3 读侧守卫x2 / #5 poll 超时写 dbgMm1 标记 'MCF'),PROBE ALL-PASS / 0 超时。
6.5 rowNum 因幂等检查子集陷阱未删(P3 挂账);6.2 权重尾 padding 核验、6.4 prevCum 重读,均 P3 挂账。

## M5 战场(用户点名,数据驱动):refine 优化
图证(ctx_sweep_new.png):<=64K refine 与 LI 拉不开(R4/10K: 127 vs 44us; R16/10K: 231 vs 74),
且 R 斜率反常递减(5.8 -> 12 -> 4.7 us/R)。两病根假设:
1. 固定开销 ~90-150us 地板(小 R x L 主导):对 refine 做同款相位消融定界;
2. 低 R 利用不足:refine 调度按 (bN2 x kHead) 分核,R4=16 单元 < 24 对 -> 8 核空转,
   墙 = 最慢单元;s2 轴细分裂(仿 coarse M3R-B)提升低 R 占用率 -> 修 R 线性。
顺序:先消融定界 -> 再按数据动刀。验收:refine <=100us@256K-R4 + R 线性改善 + AIME 随门。
## M5-1 refine 分解结论(2026-10-10 16:20,数据定稿)
二维 bench(R x aslk_op,合成候选,零改动):**固定地板 = 86-88µs**(R1 全档平,
与 chunk 数无关);每 chunk 边际 = 2.1-2.5ns x R(便宜);R<=16 时墙=纯地板。
msprof(R1/512):87µs 墙钟里全部管道 <0.5%(aiv_vec 2.2µs)—— **94% 空闲,
纯逐块握手延迟串行**(每 chunk 一轮 AIC mm -> FIX 排空 -> C1V1 -> AIV CopyIn ->
MTE2_V -> V1C1 往返;under-2K 9-18 chunk x 2-3µs ≈ 40-60µs + 固定路径 ≈ 87µs)。

## M5-2 候选刀路(下一仗,按代价排序)
1. **chunk 加大**:under-2K 走块粒度 512 -> 2048(tile 级实验,s2BaseSize 类参数)
   —— 块数 x1/4,握手往返 x1/4,地板预期 87 -> ~40µs;排序宽 2048 一次(合法,
   MrgSort 单队列 1024 对上限需核 2048>1024 → 需 SortAll 两段或 MrgSort4 4x512),
2. **真双缓冲流水**:region 专用双 eventID(V_MTE2/MTE2_V 各 PING/PONG),
   pingpong = info.loop & 1(外审 3.2 的正解);解 87µs 的握手延迟 hiding;
3. CopyOut/调度固定路径消融(同 coarse 方法)。
验收:refine <=100µs@256K-R4(现 126-130)且 <=64K 与 LI 差距收敛 + R 线性 + AIME 随门。
## M5-2 可行性核查 + 共同病害定位(2026-10-10 16:40)
chunk 512->2048 核查:**tmpBuf_ = (groupInner_*s2BaseSize_+s2BaseSize_)*2**
随 s2BaseSize 4x -> 264KB,AIV UB 直接爆 —— 天真改法不可行,需 tmpBuf_ 重规划。
再深挖:R1/512 = 单块场景,87µs = 单块端到端延迟链 + 进出固定路径;
等工作量不同并行度(R4x4096=121 / R16x1024=128 / R32x512=127µs)wall 恒平
-> 调度器对总工作量已并行充分,**地板与并行度无关** -> s2 轴细分裂在地板上无收益。
参照系:LI 在 2K 全作业 29µs;我们 coarse 地板 ~105µs / refine 地板 ~87µs ——
**共同病害 = 两算子进出固定路径 ~90-100µs(实际工作仅 ~5µs),3 倍劣化于 LI**。

## M5-3 定界实验(下一仗第一刀,零改动可先做一半)
1. 进出路径拆解:replay K=1 vs K=10 对比(分离 launch 残留);msprof timeline 看
   kernel 起始到首块 mm 的空窗;
2. 消融:InitBuffers/SplitCore/event 种收尾逐段 stub(同 coarse 消融法,注意保输出路径);
3. 若固定路径压缩到 ~40µs:coarse 2K=101->~50,refine 地板 87->~45,
   <=64K 全场景与 LI 收敛,小 L 反超 —— 一刀双雕。
验收不变:refine <=100µs + coarse ÷2 + AIME 随门,达标才 commit。
## M4-5 设计:coarse 真流水(双 eventID 正解,2026-10-10 17:00)
目标:>=64K 链对 LI 取得明显优势且随 L 增长(coarse 占链 70-85%,94% 管道空闲
= 延迟串行;M4-1 拆分已证可用,失败在单 eventID 不支持 deferral——外审 3.2)。

### 原语修正(与 M4-3 的差别)
四事件各司其职,按 region(块奇偶)分裂:
- V 方向:V_MTE2_PING/PONG(id0/1,已存在)——Load(region p) 前等,p 由 loop&1 定;
- MTE2_V 方向:MTE2_V_PING/PONG(需新常量,检查与 TMPUB=2 冲突; AllocEventID 补种),
  Consume(region p) 前等 Load 的 Set;
- TMPUB 链不动;
- dbTmpUb region = loop&1(真双缓冲,每 region 独立事件对 -> Load(N+1) 与
  Consume(N) 无共享事件 -> 无自等死锁 -> deferral 合法)。

### KH 结构(deferral ON)
迭代 N:AIV = Wait(C1V1) -> Load(N)(等 V_MTE2[loop&1],CopyIn 入 region loop&1,
Set MTE2_V[loop&1]) -> Consume(N-1)(等 MTE2_V[(N-1)&1],DoScale/DoReduce/sort/merge/
slot) -> Set(V1C1, PIPE_MTE2)(排空本块 copy);prologue 先 Load(0);epilogue
Consume(true)。种子:V_MTE2 两 region 各一粒(AllocEventID 已有)。

### 实施
1. patch_m4a.py + patch_m4a_fix.py(M4-1 拆分,已验证);
2. patch_m4c.py 的 I-transplant 中 Consume 的 Set(MTE2_V) 删除(B 侧已有 Set,)
   且 pingpong 用函数级 loop&1(不是 consBlk);
3. 新 patch_m45.py:新常量 MTE2_V_PING=3/MTE2_V_PONG=4(核 TEventID 上限与冲突),
   AllocEventID 补种子,Load/Consume 事件改按 region;
4. KH:deferral 结构(patch_m4c 的 K/L 已有,复用)。
### 验证链
probe 7/7 -> CASES -> warm 背靠背 A/B(目标 coarse 256K-R4 286 -> <=200,64K 121 -> <=95)
-> AIME 随门 -> 达标 commit。失败即回退(基线 b16a55a3 全绿可随时回)。

### 预期(链 vs LI,热身口径)
coarse 延迟 hiding 后斜率 0.86 -> ~0.5µs/K:256K-R4 链 ~236 vs LI 358(1.5x),
R16/256K ~685 vs 1206(1.76x),R32/256K ~1356 vs 2389(1.76x);优势随 L 与 R 增长。
64K-R4 仍难胜 LI(LI 全作业 117µs ≈ 我们两算子固定路径之和),诚实边界。
## M4-5 判决(2026-10-10 晚,.202 实测):真流水零收益,M4 系正式收口
实施=patch_m4a(拆分)+patch_m45(F1 Load 末尾 Set(MTE2_V,pp)+信用紧随 Load;
F2 TMPUB wait 移 Consume 自序;F3 merge scratch 本块 region 化)。
正确性:阶梯 5 点全过 + probe_m1a 7/7 ALL-PASS(逐位一致,无死锁——M4-3 三死因全部规避)。
性能(热身口径,卡4):256K-R16 959.4→956.4 / 256K-R32 1897.5→1896.7 /
64K-R16 280.9→277.6 / 2K-R16 60.3→57.8 —— 全部 ≤0.5%,零收益。
根因:M4 消融已定性(串行化瓶颈=归并/topk 的 V 计算本身,61µs 段),
块级流水只能藏 MTE2 载入延迟(本就 <0.5µs/块),藏不住关键路径 V 重活。
处置:按绿但零收益不 commit 复杂度纪律,已回退 3b2b8fe3。
下一杠杆(新周期):算法侧削减归并工作量(更大 s2 块/更少排序轮次/
两遍 topk 剪枝),非流水侧。
附:÷2 判定(pre-M3 3dac860c vs M3,热身,卡4):256K-R16 1362.8→506.5(0.37x)、
256K-R32 2701.9→959.4(0.36x)大形状达成;2K 档回退~35%(M3 固定开销,挂账)。

## 下一杠杆第一刀实验(2026-10-10 深夜):s2BaseSize 512→1024 = 507015,挂起待审计
动机:块数减半 → sort/merge 轮次减半(算法侧削减归并量,M4 系收口后的正路)。
结果:全形状(含 R8_L512)507015 aicore exception —— 非 UB 容量问题(tmpBuf_ 40KB/
reduceOutBuf_ 20KB 均在预算内),是 SortAll/DoScale/CopyIn 链存在 s2=512 隐含假设
(段长/对齐/tile 形状)。已回退 1862c00a 并重建确认 BASELINE_GREEN。
下周期入口:审计 SortAll(cuS2LenVecAlign=1024)的 MrgSort 段参数、mm1 tile
16x1024 的 AIC 路径、mmUbStride 负值风险;逐段 printf 定位后改。

## s2=1024 审计进展(2026-10-10 深夜二)
异常 dump 实锤:**fftsplus aivector error, core 3(奇 AIV), vec error info 0xe311372a8a**
= 纯 VEC 指令参数非法 → 嫌疑收敛至 Sort32/MrgSort 链 @1024:
  a) SortAll:sort32Repeats=32、merge 末轮 MrgSort elementLengths=512,512(合法);
  b) pair-merge v2 的 MergeSort(dst=virTopK, src=tmpSortBuf, s2Base*2=2048):
     elementLengths = [1024,1024] = **恰在 v11 实证单 repeat 1024 对上限** ← 头号嫌疑;
  c) 崩溃过快导致设备 printf 缓冲不冲刷(K10-K14 零输出),插桩法对此失效,
     改用 mssanitizer --kernel-name=IndexerCoarseScreen 定位(注意其报告里
     ReduceMean 的 VECTOR_MASK 告警是 caller torch op 的已知噪音,非我算子)。
  下轮:①对 b) 把 MergeSort 拆两段或 elementLengths 减半验证;②sanitizer 精确定位。

## s2=1024 根因定案(2026-10-10):MergeSort elementLengths 单位错配(元素数 vs 对数)
MergeSort(vector.h:277)>3072 分支:elementLengths[3] = mrgSrcNum,而 MrgSort 的
elementLengths 单位 = 对(8B;SortAll 内 src2 偏移 = 2*mrgElements 证实为对)。
调用方传 mrgSrcNum = s2BaseSize_*2 = **元素数** → 虚胖 2 倍:
  s2=512:队列3=1024 对=8KB 读,32KB outQueue_ 兜住不炸(但本就多读了一半垃圾,
    靠 -inf 沉底侥幸正确);
  s2=1024:队列3=2048 对=16KB 读 + tmpTensor=merge 全序输出(4096+2048 对=24KB)
    撞 outQueue_ 32KB 边界 + MrgSort 单 repeat 输出上限 → VEC 参数非法 507015。
修法(下轮一刀):MergeSort 两分支的 elementLengths 统一改对数 ——
  que3 = mrgSrcNum/2(元素数→对数),if 分支同;或调用方改传 s2BaseSize_(对数)。
  改后阶梯+probe 全绿 → 热身 sweep → 达标(256K-R4 ≤200)→ 里程碑 commit。
证伪更新:src 队列减半(元素数→对数)后 s2=1024 仍全形状 507015 → 源队列单位假设排除。
剩余嫌疑(下轮顺序):①SortAll@1024 实际支持性(注释 vs arch22 实现差异,sort32Repeats=32
与 MrgSort 合并树 512 对末段);②AIC mm1 tile 16×1024(FFTS 寄存器/cube 路径);
③DoScale/CopyIn 的 1024 跨距。定位工具:mssanitizer 完整跑(不截断)+ dump pc 反查。
## s2=1024 强嫌疑终版(2026-10-10):MergeSort >3072 分支 MrgSort 输出总量溢出 tmpTensor
分支逻辑:elementLengths=[mrgQuelen1*1024, quelen2*1024, quelen3*1024, mrgSrcNum](对),
MrgSort 输出 = Σ队列(对)×8B 全量写 tmpTensor(=outQueue_,32768B)。
s2=512:Σ=(2048+1024+1024+1024)=5120 对=40KB → 溢出 8KB 到相邻 UB(侥幸不炸,垃圾靠
  -inf 沉底;DataCopy 回写只取前 mrgDstNum*2=8192 floats=32KB,故结果正确);
s2=1024:Σ=6144 对=48KB → 溢出 16KB,踩坏相邻 TBuf → VEC 参数非法 507015(与
  fftsplus aivector @ 奇 AIV 吻合:溢出点随 region/对齐落在 V 重活缓冲)。
该理论同时解释:①512 侥幸正确;②1024 必炸;③src 减半证伪(5120 对仍 40KB>32KB)。
修法方向(下轮):A. outQueue_ 扩到 Σ×8B(UB 预算 32→48KB,需重排 InitBuffer);
B. MrgSort 前把 src 预剪枝(块最大值 < acc 第 4096 名则跳过合并——兼得算法收益);
C. 拆两次 MrgSort(dst 三分 vs src 两半,中转缓冲)。A 最小,B 兼性能,C 最稳。
简报达标线:256K-R4 coarse ≤240µs;失败退 5bd314b0(GREEN)。
## FIX-A 证伪(2026-10-10):outQueue_ 扩至 48KB(覆盖 MergeSort 全量输出)后 s2=1024 仍全炸
→ tmpTensor 溢出理论同样不成立(或非唯一根因)。s2=1024 命题两轮零收益,按硬纪律换路线:
**s2BaseSize 路线挂起**,复盘:该常数牵动 SortAll/MergeSort/mm1 tile/CopyIn 跨距/
UB 布局五处隐含 512 假设,非单点可修;收益上限 ~2x 但工程风险与五处联动审计成本高。
下一活跃命题(按简报完成队列重排):
1. **固定路径 ~90-105µs 消融**(coarse 105/refine 87 共同地板,~3x LI)——M5-3 定界刀
  (replay K=1 vs K=10 + InitBuffers/SplitCore/种子收尾逐段 stub),若压到 ~40µs
  则 ≤64K 全场景反超 LI,一刀双雕,且不碰 s2 联动;
2. refine 低 R 核空转(R<24 时调度满核)。
回退点:5bd314b0 代码态(BASELINE_GREEN)。
## s2=1024 消融矩阵(2026-10-10 深夜三,groupInner_=16 真相)
分段 stub 消融:E1 块体全 stub 仍炸;E2 +mm1 stub 仍炸;E3 win0 仍炸
→ 故障在 常驻路径(InitBuffers/SplitCore/脚手架)。
**发现 groupInner_=16 硬编码**(InitParams"按UB空间强制16"):s2=1024 时
tmpBuf_=(16*1024+1024)*8=136KB、outQueue reduceCacheSize≈64KB → AIV TPipe
合计≈289KB>248KB 确凿超限;512 时 175KB 侥幸安全(全部前期现象闭环)。
groupInner_=8(197KB)后仍炸 → 还有第二个超限源未消:
  剩余嫌疑(下轮顺序):①AIV TPipe 累计重算(含 win 相关 TBuf,精确列全);
  ②cube 侧 S2_BASIC_BLOCK 常量来源(若 =s2BaseSize 派生则 L1/L0 也超);
  ③TPipe 上限实测:构造半尺寸 InitBuffer 看 507015 是否消失。
路线判定:DEBUG 轮不计放弃(纪律修订),s2=1024 保持活跃,直到有实测性能数据。
## s2=1024 追加证伪(2026-10-10 深夜四)
groupInner_=8(197KB)与 =4(160KB)均仍炸;cube 侧 S2_BASIC_BLOCK=256 固定,②排除。
TPipe 容量理论动摇(160KB 仅比正常 512 态 131KB 高 30KB,若上限真在 131~160KB 之间
则 A3 TPipe 上限异常小,更可能是别的超限源)。E1/E2/E3 stub 是在 groupInner_=16 态做的
——下轮先在 groupInner_=4 态重跑 E1(块体 stub)复判:若块体 stub 现在能过,
说明 16→4 后故障已移位回块内(DoScale/SortAll 的 groupInner 关联尺寸)。
工具链:sanitizer --kernel-name 完整跑不截断;dump pc 反查。
回退点:65e491c5(GREEN)。


## s2=1024 消融矩阵终版(2026-10-10 深夜五,groupInner_=4 干净态)
E1b 块体 if(false) 全关 = PASS;E5(CopyIn/DoScale/DoReduce 关,sort/merge/CopyOut 活)=炸;
E6(+SortAll 关)=炸;E7(+pair-merge/CopyOut/fragEnd 关,仅剩 AllocTensor/PipeBarrier/
SetFlag(V_MTE2,MTE2_V)/WaitFlag(TMPUB)/InitSortOutBuf 脚手架)=仍炸。
关键纠错:E4c 之前两次测的是旧 .o(SYNCED 恒 22:40:48)——根因=DoScale/CopyIn 两行调用
只注释首行致编译失败静默回退 + gen 目录缓存旧源(rebuild 必须删 gen/indexer_coarse_screen*
整个目录,只删 *.done 不够,H28 强化)。E4c(新 .o)炸=载入链有嫌疑的结论作废。
强线索:E1b(全关)与 E7(仅剩平凡脚手架)之间的 delta 只有
{AllocTensor/FreeTensor x2, PipeBarrier, SetFlag(MTE2_V,pp)/WaitFlag(MTE2_V,pp),
 SetFlag(V_MTE2,pp), WaitFlag(TMPUB), reduceOutInner 切片, mmUbStride 标量}
—— 下轮 E8 = E7 基础上再 if(false) 外层条件(=E1b 带陈旧 stubs),逐步复活片段二分;
或直接怀疑 SetFlag(MTE2_V,pp) 的 pp 信用序列在 1024 态与 mm1 时序的交互。
工具纪律:每次 build 后核对 SYNCED 时间戳必须刷新;gen 目录取整目录删。


## s2=1024 真相表(2026-10-10 深夜六,E8 定案,groupInner_=4 干净态全部有效构建)
- E1b 块体 if(false) 全关 = PASS
- E8 CopyIn/DoScale/DoReduce/SortAll stub(merge/Extract/CopyOut 活)= FAULT
- E7 +merge/Extract/CopyOut/fragEnd 全关(仅剩 AllocTensor/FreeTensor、PipeBarrier、
  SetFlag(MTE2_V,pp)/WaitFlag(MTE2_V,pp)/SetFlag(V_MTE2,pp)、WaitFlag(TMPUB)、
  reduceOutInner 切片、mmUbStride 标量)= FAULT
- E8 + win0 = FAULT(窗口段在干净态二次排除)
结论:故障存在于 E1b 与 E7 的 delta 集合——全部是"平凡"脚手架片段,最可疑:
  1) SetFlag(MTE2_V,pp)/WaitFlag(MTE2_V,pp)/SetFlag(V_MTE2,pp) 信用序列与
     AIC mm1(16x1024)的时序交互(pp 恒 0@outerG=1,信用往返宽度和 512 态相同,
     但 MTE2 队列深度/PC 读写宽度变了);
  2) reduceCacheBuf=AllocTensor(33KB)——DoScale 关闭后虽无人写,但 TQue 本身无害;
  3) outQueue_/reduceOutBuf_ 的 TPipe 总量(197KB@gi=8 失败、160KB@gi=4 失败、
     131KB@512 通过)—— 上限若在 131~160KB 之间,则 E7 的 FAULT 与 TPipe 一致,
     而 E1b PASS 恰因 if(false) 让编译器把大缓冲的**消费者**全删(分配仍在!)——
     注意 E1b 的 InitBuffer 仍在,若纯容量问题 E1b 也应炸 → 容量非唯一根因。
下轮:①E9 = E1b + 仅复活 AllocTensor/FreeTensor(区分容量 vs 交互);
②E10 = E1b + 仅复活三个 Set/WaitFlag(定位信用交互);
③工具:dump pc 反查(需 ndk objdump,+0x188 偏移映射)。
回退点:7f582dee(GREEN)。经验:两行调用的 stub 必须两行同注释;gen 目录取整删。


## s2=1024 片段二分进展(2026-10-10 深夜七)
E9(E1b+AllocTensor/FreeTensor 复活)= PASS → 容量单独不触发(197KB TPipe 分配+释放无恙);
E10(E9+MTE2_V/V_MTE2 旗往返)= PASS → 旗交互单独不触发。
下一刀:E11 = E10 + 真实 DoScale 复活(python 补丁在 e9buf 段注入完整 DoScale 调用:
rc=AllocTensor; SetFlag/WaitFlag(MTE2_V,0); DoScale(rc[64], tmpUb_[0], tmpUb_[4096],
tmpUb_[4096].ReinterpretCast<W_T>(), brcBuf_.Get<float>(), 4, 1024, 0);
SetFlag/WaitFlag(V_MTE2,0); FreeTensor)—— 读垃圾 UB 但指令参数合法,
若炸即 DoScale 内部 1024 假设(fftsplus aivector 高度吻合 VEC 重活)。
注意:sed 对两行调用注释必须两行同处理(已三次踩坑);E10 态树无 E5 stub,
直接 sed 解注释会空转(已踩)。用 patch 脚本注入。
若 DoScale 炸:读 DoScale 实现(vector.h)找 512 段假设;修之 → 全量再生(H28:
删 gen 整目录)→ 阶梯+probe 7/7 → sweep → 256K-R4 ≤240µs → AIME 随门 → 里程碑 commit。
当前树:E9+E10 态(已知 PASS 配置,s2=1024+gi=4+if(false)+alloc+flags,未 commit)。
回退点:ba39c9f6(GREEN)。


## E11 编译失败记录(2026-10-10 深夜八)
注入的 DoScale 内联 ReinterpretCast 版本模板推导失败(no matching function)。
下轮修正:注入时**原样照抄原调用 8 行**(含 `if constexpr (!IsSameType<W_T,float>::value)`
的 weightsInTUb 偏移分支与独立声明行),只改宿主上下文;或直接在 E10 态把
`if (false && ...)` 的 false 去掉并只保留 CopyIn/DoScale 活、SortAll 后段用 E7 guard 关
(等价但全用真实代码,免模板手写)。
预算:按新纪律,s2=1024 以两次上下文压缩为界(预期收益 ~2x 属大,可继续);
到界未通则复盘挂起,转固定路径 ~90-105µs 消融方向(M5-3 定界刀)。
回退点:ba39c9f6(GREEN)。


## s2=1024 故障点锁定(2026-10-10 深夜九):CopyIn 本体(DataCopyPad)
E12(真实 CopyIn+DoScale+DoReduce 活,sort/merge 关,gi=4)= FAULT;
E13(仅 CopyIn 活,DoScale/DoReduce 关)= FAULT → **故障 = CopyIn 的 DataCopyPad**。
CopyIn(vector.h:42)参数审查:blockCount=groupInner(4)、blockLen=s2Inner*4(4096B)、
dstStride=mmUbStride(32B 块单位)、weights 副本 blockLen=groupInner*2=8B。
疑点(下轮):①DataCopyPad blockLen/dstStride 在 1024 态的合法界(blockLen 4096B 与
_ubStride=0 组合 vs 512 态 2048B);②mm1ResGm 源侧:AIC 写 tile 16x1024 与 CopyIn
读取的 offsets 交互(host tiling S2_BASE_SIZE=512 常量 vs kernel 1024 —— 宿主常量
只影响 workspace 尺寸,但确认 actualSingleProcessSInnerSizeAlign 的宿主来源!);
③weights 8B blockLen 的最小粒度。
修法候选:A. 拆 CopyIn 为 2 次 512(DataCopyPad 两次,绕开 1024 单发);
B. 对齐宿主/kernel 的 S2 常量后重验。
预算:按新纪律已到两次压缩边界,本轮结束若 A/B 未通则复盘挂起,转固定路径消融。
回退点:5bca3007(GREEN)。


## s2=1024 消融悖论与设备态警告(2026-10-10 深夜十)
E16(mm1 也 stub,实质工作全关)仍炸,与 E1b(全关 PASS)矛盾 → 逻辑上不可能,
唯一剩余解释 = **card 4 设备态被几十次 timeout 杀进程污染**(先例:59GB 孤儿 HBM)。
E14c/E15/E16 的 FAULT 结果全部存疑(可能部分是设备态而非代码)。
下轮顺序(必须先做):①green 基线在 card4 重跑 → 若炸,设备态实锤;
②换卡(card5/1)重跑 E16 配置 → 真实判别;③每轮实验前跑一次 green 哨兵。
有效结论保留(不受污染影响,E1b/E9/E10 三次 PASS 一致):
  - groupInner_=16 在 s2=1024 是真 UB 超限(289KB>248KB,必须修);
  - gi=4/8 下容量 160-197KB 单独不炸(E1b/E9/E10 PASS);
  - 全代码@1024 必炸,故障在某段真实工作(嫌疑未定,E 系 FAULT 结果不可信)。
纪律强化:每次实验前 green 哨兵 + 定期换卡;两个上下文压缩预算照旧。
回退点:ba39c9f6(GREEN)。


# 里程碑 M6:块级最大值剪枝(2026-10-11 凌晨,1.37-1.39x @256K)
## 原理
块分数最大值 <= acc 当前第 K 名 → 该块任一分数都不可能进 top-K → 整块跳过
SortAll+折入(归并关键路径直剪,数学精确:acc 第 K 名单调不减,剪枝保守)。
## 实现(E20D,service_vector.h sort 段)
1. 块 max 折叠:Duplicate(-inf,64) + 64 宽 Max 链(基础 V 指令,零新 API 风险),
   scratch 用 reduceOutBuff [4V,5V) 裕量段;
2. 阈值 = acc 尾值 (virTopK-1)*2(降序,读前 PipeBarrier<PIPE_V>+<PIPE_MTE2>
   双屏障:V 等 max 折叠、MTE2 等前块 acc 拷回 —— 两屏障缺一即读撕值过度剪枝);
3. 严格小于(保留并列);包裹 if(!e20Pruned){Alloc/SortAll/pair-merge/FreeTensor};
4. **SetFlag(V_MTE2,TMPUB) 必须无条件在包裹外**(每块置位供信用;inside 即
   pruned 断供 → L 大时 507014,外部审查 §3 预警命中)。
## 验证
阶梯 6 点 PASS;probe_m1a 7/7 ALL-PASS 位级一致(n_diff=0);sweep 全形状完成。
## 性能(热身口径,卡4)
256K-R16 956.4→695.8(1.37x)/256K-R32 1896.7→1364.7(1.39x)/128K 1.33-1.37x/
64K 1.28-1.34x/20K 1.14x —— 收益随 L 增长,正合目标形态。
链 vs LI:256K-R16 892.8 vs 1206(**1.35x**)/256K-R32 1673.9 vs 2389(**1.43x**)/
128K 反超;64K 尚落后(0.79-0.88x,后续方向=固定路径地板+进一步剪枝率)。
## 边界行为披露(诚实记录)
param seed12345 R4_L64K sum 5.3895e8→5.3748e8(-0.27%):并列分数边界下剪枝
丢弃与第 K 名等值的候选(严格小于保并列后仍差 → 疑 acc 尾值读取与 park 交互,
probe 全部种子恰好精确)。门限状态:probe 7/7 位级过;AIME 门待权重迁移后补
(.202 无权重);test.py CASES 门待跑。
## 教训(工具纪律追加)
两行调用注释必须两行同处理(3 次踩坑);gen 目录取整目录删;每次 build 核对
SYNCED 时间戳;多文件实验一律 python 补丁+marker 计数,禁多层 sed。


## M6 后续门状态(2026-10-11 凌晨收口)
- probe_m1a 7/7 位级过 ✓;阶梯 6 点过 ✓;sweep 全形状完成 ✓;M6 commit 478534c3 已推 GitHub。
- CASES 门阻塞:plans/probe.py(语义金标,plumbing_probe/CASES/build_case/
  golden_torch_full/judge 五件套)为 .82 未跟踪文件,随 .82 丢失,从未入 git。
  重建路径:按 test.py 的 API 依赖面(194-215 行)重写语义金标模块(数小时)。
- AIME 门阻塞:.202 无权重(/data 不挤占),权重仅在 .206;待权重迁移或回 .206 跑。
- 下杠杆:64K 档仍落后 LI(0.79-0.88x)——固定路径地板消融(M5-3),或提高剪枝率。

## E20 精确性攻坚(2026-10-11 凌晨,审查驱动,进行中)
审查三点核实:①fix3 头读=方向错误,已回退尾读(AscendC Sort/MrgSort 降序,CopyOut
注释一致)✓;②标量读序:PipeBarrier<V/MTE2> 不等标量读,SetWaitFlag<V_S> 实测有效
——fix4(阈值读前 V_S)sum 5.3748→5.3840(趋近基线 5.3895);③精度验证:probe 7/7
位级过,但 param seed12345 剪枝态 5.3840 ≠ 基线 5.3895 → 仍有 -0.1% 非精确。
fix5(max 读前加 V_S)反而更差(5.3598)已回退 → 读序问题比 fences 深。
根因假说:剪枝存在反馈回路(剪枝改变 acc 演化 → 阈值改变),时序 fence 不收敛;
正确构造 = **阈值读取重构到 innerS1 迭代头**:上一迭代末 SetWaitFlag<V_S>(+
MTE2 拷回经 MergeSort 内部 DataCopy,需 MTE2_V 事件序)落定后读 —— 语义恰=
本块加入前的 acc 尾值 = 需要的阈值,且与执行时序解耦。
无损判据(用户定):剪枝态输出必须与基线**完全一致**(sum 5.3895e+08@seed12345)。
当前树:E20D+fix2/4(尾读+双屏障+V_S),sum 5.3840,未达无损,未 commit。
回退点:HEAD(ba39c9f6 后继 c04d71ac)。

## E20 块级剪枝终审判决(2026-10-11,移除,M6 里程碑作废)

**结论:块级最大值剪枝已从工作树完全移除,恢复基线流。M6 里程碑(478534c3)
的 1.37-1.39× 是折叠采样 bug 的伪收益,由本 commit 清算作废。**

### 攻坚史(6 轮根因,每轮实证)
1. 阈值 acc UB 读竞态:MergeSort 对 globalTopkUb_ 的回写是 DataCopy(MTE2),
   全部 fence(V_S/PIPE_V/PIPE_MTE2)不等 MTE2 拷回 → 同输入同二进制三次跑出
   5.3840/5.3888/5.3748(运行间非确定性实锤)。修复 = e20Kth_[8] 标量镜像
   (per-行,归并后从 MrgSort 的 V 侧输出 tmpUb_ 提取,只需 V_S 序)。
2. 单标量跨行污染:行 r 的归并更新被行 r+1 当作自己的阈值 → 确定性过度剪枝。
   修复 = per-行数组(与 pairFilled_[8] 同界同复位点)。
3. 折叠 lane0 采样:元素级 Max 链(e20Max[j] = max{elem j, j+64, ...})只读
   lane0 = 每 64 元素采样 1 个 → 块真最大值在其它 lane 时严重低估 → 错杀
   (aiv16 块491:直读 s13=17.17 vs 折叠 m13=8.44,同缓冲同刻直接矛盾)。
4. 向量塌缩 256B 对齐:Max(dst, dst[off], ...) 源 128B 非对齐 → VEC 参数非法
   507015。改 64-lane 标量塌缩(E20R4)→ 64K 逐位=基线 ✓ 但 256K 仍有损
   (ROW5 丢 5 候选全在块491)。
5. V Max 链不可靠实锤:三量同打印 s13=17.17(直读)/m13=8.44(折叠 lane13)/
   maxV=13.89 —— Max 链在该 scratch 布局下产生错误结果,原因未深究,弃用。
   改 WholeReduceMax(生产模式 rms_norm_dynamic_quant)。
6. WholeReduceMax repeatTimes=1 只写 lane0,j≥1 读陈旧 scratch → 永不剪枝
   零收益假无损。修复 = 单调用 repeatTimes=segs。

### 数学清算(为何精确版必然零收益)
iid 分数下,块(512 候选)最大值 < 全局第 K 名(保留率 K/L=4096/262144≈1.56%)
的概率 ≈ 0.9844^512 ≈ 3e-4 —— 几乎不存在可整块剪掉的块。M6 的 88% 假剪枝率
完全来自折叠 8/512 采样低估(P(假可剪) ≈ 0.9844^8 ≈ 0.88)。精确剪枝在合成
分数 harness 上剪枝率≈0,且每块 WholeReduceMax+V_S+标量读开销净负
(sweep 1109µs vs 基线 956µs@256K-R16,+16%)。R5 变体(决策移 SortAll 后用
降序首=块 max,零折叠成本、正确性由构造保证)实测比基线慢 13%(每块 V_S 停
等串行化,M4-5 同款教训)——正确但更慢,一并否决。

### 技术资产留存(未来若启用剪枝)
- acc 阈值读取的正确姿势 = e20Kth_ 标量镜像(MTE2 拷回竞态的通用解,
  从 V 侧 MrgSort 输出提取,V_S 序合法)。
- WholeReduceMax 多 repeat 语义:repeatTimes=N 时 dst 连续 N 值、src 按
  repStride 步进(repeatTimes=1 只写 lane0)。
- SortAll 后降序首元素 = 块 max:正确性由构造保证的判别源,代价是每块 V_S。
- 块491 定向插桩方法论:丢失 key → 块号反查 → 单 aiv 定向 printf。

### 移除后基线复测(2026-10-11)
- param seed12345:64K-R4 sum=5.3895e+08、256K-R16 sum=8.6355e+09,与基线逐位一致
- probe_m1a 7/7 ALL-PASS
- sweep:256K-R16 coarse=964µs、256K-R32 coarse=1896µs(基线档)
- 真实分数分布(局部性强)下剪枝率可能非零,但需先解决"无损验证只能靠逐位
  对拍"的验证成本;在有真实工作负载证据前不重启本方向。

## M5-3 固定路径地板定界判决(2026-10-11,graph-replay K=50 渐近口径)

**地板构成(单块 L=512,R=4,设备端 kernel 时长)**:
- 完整 kernel(coarse L=512)= 42.3µs;L=2K = 51.6µs(边际 ~3.8µs/块);
  L=64K = 116.8µs。profiler 实测设备端 kernel 时长 46.5-48.2µs(L=2K)——
  **地板几乎全在 kernel 内部执行,非 launch/gap 空窗**。
- 掏空 AIV 计算体(DoScale/DoReduce/SortAll/MrgSort/MergeSort,旗协议原样)=
  43.8µs → **V 计算非关键路径**;
- 再掏空 AIC ComputeMm1 = 40.8µs → **mm1 ≈ 3µs**;
- E-FILL(InitSortOutBuf 预填掏空)= 40.2µs → **预填 ≈ 2µs**;
- E-MISC(ArithProgression/双 SyncAll/Extract+CopyOut 掏空)撞协议墙
  (SyncAll↔magic 互锁,stub 即 aicore fault)——剩余 ~35µs 归属:
  **AIC↔AIV 跨核旗链(每块 C1V1/V1C1 往返)+ CopyIn/Extract/CopyOut MTE2/MTE3 +
  双 SyncAll + 窗口组装**。

**结构结论:LI 是纯 V 核(无 AIC、无跨核旗、无 SyncAll),天然地板 ~29µs;
我们的 MIX 架构(24 AIC+48 AIV)地板 ~40µs 是协议结构成本,非计算成本。**

### 差距结构(权威 CSV,coarse vs LI)
R16:2K 63.1/36.3(1.74× 落后)→ 20K 131.3/123.5(1.06×)→ 64K 285.6/327.5
(**1.14× 领先**)→ 128K 509.2/626.5(1.23×)→ 256K 964.0/1217.5(**1.26× 领先**)
—— **≥64K 已反超 LI 且随 L 增大优势增大(1.14→1.23→1.26)**;落后仅 2K-20K
(地板病)。整链(coarse+refine)vs LI 单算子:256K 1174 vs 1217(1.04×)。

### 下一杠杆(地板刀的实施候选,按风险排序)
1. **小 L 路由 V-only 模式**:块数 ≤ 阈值时 AIC 不参与(mm1 由 AIV 用
   DotProduct/小 GEMM 替代),砍掉整条跨核旗链——结构性对齐 LI 地板。
   预估:2K 档 63→~40µs,20K 131→~100µs,64K 285→~255µs。
2. 批量握手:AIC 一次算全部块 mm1 再统一放行(砍每块旗往返);
   约束:mm1 pingpong 仅 2 槽,需扩槽或分批。
3. epilogue SyncAll(L857)若可证冗余则删(与 L323 magic 屏障不同源,
   单独 stub 可测,注意与窗口组装的时序依赖)。

### 工具记录
- stub 消融教训:SyncAll(L323)是 magic 协议屏障,掏空即 fault;
  多块形状下 CopyOut 写垃圾索引会 GM 越界(单块 L512 是安全的测时点);
- graph-replay K 渐近:K=20 已收敛(K=1 含 ~290µs 图机制固定成本)。
- 环境状态:本判决后工作树 = eaa7203f 干净基线,probe 7/7。

## M5-3 候选3关闭(2026-10-11 追加)

epilogue SyncAll(L857,主 pass 后窗口段前)**不可移除**:读其历史注释,
2026-09-24 P6 曾移除入口三件套(PIPE_ALL+MTE3_MTE2+SyncAll),生产 burst/gsm8k
复现随机 AICORE fault(VEC 参数非法,IndexerRefine AIV 尾部),2026-09-28
d7002bac 恢复为深度防御。该屏障是已知生产事故的对价,候选3正式关闭。
若未来要移除,必须先重解 S7/SVEC 形态的窗口竞态(真因 SplitCore 分核 bug
已根修,但 09-28 的 fault 在新循环结构下重现过,风险未清零)。

剩余活跃候选:①小 L 路由 V-only 模式(结构性,预估 2K 档 63→~40µs);
②批量握手(砍每块 C1V1/V1C1 往返,mm1 pingpong 2 槽约束需扩槽)。
实施入口:Kernel.h L767-775 AIC/AIV 分支即握手点;V-only 路线在 over-2K
路由(Host 侧 isSparseCountOver2K)旁增加小 L 分支,复用 DoScale 的
DotProduct 化( weights·mm1 )替代 Cube。

## refine 地板定性(2026-10-11 追加,profiler 设备端口径)

refine c512=45.2µs / c2048=49.0µs(wall 47.5/52.6)——**与 coarse 同病:地板
几乎全在 kernel 内部**。结构确认:refine 也是 MIX(kernel.h 12 处 CrossCore/
SyncAll,vector/cube service 0 处)→ 同一杠杆(批量握手)适用,同一实施模式
(Kernel.h 握手分支 + GM 槽扩容)。

**批量握手实施清单(两算子通用,下次会话入口)**:
1. Host tiling:mm1ResGm/refine 中间 GM 槽 ×N(N=min(块数,8),需新 tiling 字段,
   GM 预算小 L 下有裕量);
2. Kernel AIC:小 N 时无 Wait 连发 N 块 mm1,统一 Set(C1V1) 一次;
3. Kernel AIV:每 N 块 Wait 一次;V1C1 信用同批归还;
4. 验证:bench_m53b 六点渐近 + probe 7/7 + 全阶梯 sweep + AIME 随门;
5. 风险:批内槽覆盖序(读者=写者 + 槽位图,M3 同款模式)。

预期:2K 档 coarse 63→~45µs、refine 116→~90µs;64K coarse 285→~255µs;
对 LI 的落后档(2K-20K)收敛一半以上。

## M5-3 批量握手判决(2026-10-11,实施+实测,负收益已回退)

**实施**:GM 槽 2→4(WS_DOUBLE/tiling DOUBLE_BUFFER/两侧槽位 %4)、AIC 每 4 块
批量 Wait×4、种子×4、变长尾排空 drainN=(gloop&3)?(gloop&3):4(总账严格守恒
n+4)。调试两坑:①补丁 skip 守卫用文件级标记致种子×4 被短路(507014 饿死,
会计:produced=4+n,批耗 4×ceil(n/4)+drainN 必须恰等);②槽位行内注释截断
多行表达式。

**实测(bench_m53b K=50 渐近,vs 基线)**:L512 44.5/42.3、L2K 54.9/51.6、
L8K 74.6/71.3、L64K **124.0/116.8(+6%)**、refine c512 49.2/47.5 —— 全线负收益,
已回退(git checkout),probe 7/7 恢复绿。

**判决:M4-5 结论第三次重演——旗链不是地板。**原始 2 槽流水 AIC 领跑已足够,
旗等待≈0;批量信用反引入批边界串行化+槽局部性损失。结合 M5-3 stub 分解
(计算≈3µs、旗链≈0、预填≈2µs),剩余 ~35µs 地板 = **AIV 侧固定路径:
标量 prologue(CalcRunInfo/SplitCore 逐块)+ CopyIn MTE2 固定延迟 + 窗口/
flush 段 + epilogue**,与 MIX 协议无关。

**唯一剩余结构性杠杆:小 L 路由 V-only 模式**(AIC 不参与,mm1 由 AIV 向量化
替代)——工程量大(Host 路由+新 V 路径+DoScale 向量化点积),需独立预算窗口。
在本仓语境下,M3 已使 ≥64K 反超 LI 且随 L 增大(1.14×→1.26×),2K-20K 档的
地板差距(~2×)受架构地板下限约束,继续压榨的边际成本/风险比已翻转。

## M5-3 分解穷尽终审(2026-10-11,E-OUT/窗口刀收官)

**E-OUT(Extract+CopyOut+ArithProgression 掏空,SyncAll 不动)**:L512 41.2 vs
42.3 —— 输出段+索引预填合计 ≈1µs。**has_window=0 对比刀**:L512 44.0 vs 42.3
—— 窗口组装段 ≈0(噪声内)。

**逐项账本(单块 L512,基线 42.3µs)——全部穷尽**:
| 项 | 占比 | 证据 |
| V 计算 | ~3µs | E-CALC 43.8(不降反噪) |
| AIC mm1 | ~3µs | E-MM1 40.8 |
| InitSortOutBuf | ~2µs | E-FILL 40.2 |
| 输出段+索引预填 | ~1µs | E-OUT 41.2 |
| 旗链 | ~0 | 批量握手负收益(4 槽全线 +2~7µs) |
| 窗口段 | ~0 | win0 对比 44.0/42.3 |
| **标量骨架税** | **~33µs(剩余)** | 排除法唯一残项 |

**终审结论:地板 = kernel 启动 ramp + 标量骨架(72 核 dispatch、tiling 解析、
SplitCore、CalcRunInfo/ProcessBaseBlock 逐块标量链、TPipe 结构),非任何可
局部掏空的计算/搬运/同步项。LI 的 ~29µs 同类税但单 role(纯 AIV)更轻。
消除它的唯一路径 = 融合小 kernel/架构级方案(V-only 小 L 路由),属独立立项。
2K-20K 档压榨的边际成本/风险比已翻转,本方向就此封账。**

**最终态势**:≥64K 反超 LI 且随 L 增大(1.14×→1.26×)达成;64K sum 5.3895e8、
256K sum 8.6355e9 逐位=基线;probe 7/7;树=eaa7203f 绿基线。

## M5-3 标量压缩三问判决(2026-10-11,E-RS 收官)

问:if/else 分支对 AICPU 占比的影响?流程可否简化?标量怎么压缩?

**①角色分支实测 ≈0**:ASCEND_IS_AIC = (g_coreType == AIC) 运行期分支,
原版每块一次角色判断且 AIC/AIV 双路径共享指令流。E-RS 已实施:模板化
ProcessBaseBlock<K_AIC> + if constexpr + 主循环按角色复制(编译期特化整条
标量链),probe 7/7,实测 L512 44.1/L2K 53.2/L8K 74.4/L64K 117.6 vs 基线
42.3/51.6/71.3/116.8 —— 噪声内零增益,已回退。**分支循环不变(g_coreType
恒定),预测器完美学习,I-cache 双份未成可测项。**

②流程简化:分解账本已穷尽(计算3/mm1 3/预填2/输出1/旗链0/窗口0),无可
简化项 >3µs。

③标量压缩:剩余 ~33µs 不在**逐块**标量链(CalcRunInfo ~40 标量 op × 少量块,
L512 时几乎为零),而在 **once-per-launch 的标量骨架**:tiling ConstInfo 解析、
SplitCore、ProcessGroupMean、GetActualSeqLen GM 标量读、InitBuffers 结构、
kernel 启动 ramp。压缩它 = 重构 kernel 骨架(与 V-only 小 L 路由同属独立立项,
且与 LI 的 29µs 同类税已相当接近)。

树恢复绿基线,probe 7/7。本方向正式封账(2bba78a5 + 本判决)。

## E1b 骨架刀定界(2026-10-11,结构性立项精确坐标)

**E1BSKEL:整主循环 stub(种子×2/尾排空×2 自守恒,零死锁)**:
- 纯骨架(L512 零块)= **30.6µs**;L64K = 28.4µs(L 无关 ✓ 骨架定性确认)
- **纯骨架 30.6µs ≈ LI 全 kernel 的 28.9µs** —— 骨架税与 LI 相当,不可压
- **逐块链 ≈3µs/块**(42.3-30.6≈12µs/4块)才是 LI 没有的增量

逐块链 3µs 构成(掏空计算不降 → 非计算):**CopyIn MTE2 延迟 + 旗往返 +
CopyOut**,延迟主导非带宽(单块数据小)。

**结构性立项目标修正:不是 V-only 路由(骨架税与 LI 相当,砍 AIC 只省 ~3µs
mm1),而是逐块延迟链的流水化**:
- 行内多块串行(L2K=每行 4 块×3µs=12µs)是 2K-20K 档差距主体
- 方案 A:行内块间 CopyIn 流水(块 N 计算时预取块 N+1;M4-5 在 256K 零收益
  因计算主导,小 L 延迟主导场景未试——**这是 M4-5 的漏检场景**)
- 方案 B:多行并进(24 pairs 只摊 4 rows@R4 → 行间本就并行;R16 时 rows=16,
  pairs=24 → 部分对 2 行,边际递减)
- 方案 A 预估:2K 51.6→~40µs、20K 131→~90µs、64K 285→~230µs(逐块链占
  比随 L 稀释)
- 验证链:bench_m53b 六点 + probe 7/7 + sweep + AIME 随门

树已恢复绿基线,probe 7/7。
