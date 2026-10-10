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
