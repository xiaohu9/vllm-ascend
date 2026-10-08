# coarse_screen 每行扫描效率劣化 LI 19× —— 数据、定位与优化设计

> ## 2026-09-30 深夜: 均值融合的可行性终审(修正先前判断)
> 先前判断"复用 prefill 的 M1-proxy 基础设施"有误 —— 核实结果: **prefill 的
> q_bar 同样是 python 侧计算**(`index_add_` 均值, pivot_indexer.py:195-197),
> kernel 内无现成 proxy 计算设施可复用(服务文件里的 M1 组均值注释是已裁剪的
> 历史机制)。
> **均值融合的准确定性**: op 签名波及六层(def/adpt/host/tiling/kernel/caller)
> + kernel 内新增 AIV proxy 预阶段 + workspace/sync —— **完整的 M1a 类手术**,
> 与两级 GEMM 同批次实施(同一签名改造, 两件事一次做)。
> 今晚不再单独实施; 与主力 M1a 汇合后统一推进。

> ## 腾讯 LI 优化文章研读结论(支线 agent 全文抓取, 2026-09-30)
> 文章 = DeepSeek-V3.2 LI 算子概念解读(引用 cann-recipes-infer 开源实现,
> KERNEL_TYPE_MIX_AIC_1_2 = 1 Cube + 2 Vector 混合核, 与我们 coarse 同构)。
> **设计原则与我们差距的直接印证**:
> 1. **"中间分数不落 GM"** —— LI 的 cube 打分直接片上交给 vector 排序;
>    我们的 mm1Res 走 fixpipe→GM→AIV 往返(63% 瓶颈) = 第一差距;
> 2. **tile 局部 Top-K + 运行阈值前置过滤**(低于全局第 K 大的新分数不进排序)
>    = AIV 80% 排序人口的压缩方向;
> 3. **双缓冲分数 A/B 三段流水**(算 n/筛 n-1/预取 n+1)= AIV 串行占比的削减;
> 4. **ReLU 前置剪枝**、**量化融合 mm**、**tilekey 模板去运行时分支**。
> 完整借鉴清单与来源链接已存支线报告(见对话记录)。
> **注意**: 文章为概念级(无微基准数据), 深入实现须读 cann-recipes-infer 开源仓
> 的 LightningIndexer kernel 源码(本轮 GitHub/gitcode 抓取失败, 待网络)。

> ## 2026-09-30 晚: 双累积归并实测性能回退, 已回退保 357µs 态
> v11 双累积(refine 移植, 两次 2048-dst 2-list 归并替代单次 4096 宽 3-segment):
> 精度门 PASS 但 **bench 回退 ~4%**(377 vs 357, @256K 1421 vs 1350) —— 两次
> 2048 宽 2-list 归并的计算量 > 单次 4096 宽 3-segment MrgSort4。其换来的
> 相邻 swap 修正对 coarse 是 tie 级(我们的 tie-aware 门本就 PASS), 不值。
> 已 git checkout 回 357µs 态。**3-segment 的 ~50% 数据相关 swap 隐患留给
> 主力评估**(refine v11 注释的发现, coarse 同路径暴露同风险, tie 级)。

> ## 事实核查: chunk 为什么是 512 + 归并频率的真正限制(2026-09-30)
> 三条硬件约束链的交点(全部实测):
> 1. **L1 512KB**: KEY 三缓冲 3×(512×128×2B)=384KB + Q 双缓冲 64KB = 448KB。
>    512 = KEY_BUF_NUM=3 下最大 S2 块; 1024 需 KEY_BUF 3→2(预取深度减半,
>    mte2 已 46% 会恶化); 2048 = 1.5MB 实测崩。
> 2. **MrgSort4 单队列上限 1024 对**(实测 1024 可/2048 fault)。
> 3. **UB 零余量**(审计 193.5KB≈192KB 上限): 延迟槽只能挖 reduceOutBuf 保留段。
> → acc 归并频率 = L/V 是约束链的结果; pair /2 是当前 UB 布局的极限。
>
> **GM 槽变体**(下一轮可试): 延迟槽下沉 candidatesWs 段旁(需 host workspace
> += 4行×1024对×8B=32KB, op_host 改动→需一轮全量构建); kernel 侧
> DataCopy UB→GM(MTE3)+GM→UB(MTE2) + 三管道事件同步。代价每对 ~2µs
> GM 往返, 收益 acc 归并频率 /3(三槽 GM 空间充足)。净收益需实测——
> 事件三管道同步是此前三次 fault 的雷区, ID2 经验(选空闲点+双向成对)可复用。

> ## ⚠️ 断网中断状态快照(2026-09-30 ~16:30, .82/.218 双主机 ETIMEDOUT)
> **服务器树状态(未提交!)**: service_vector.h = Cycle-X2 中间态(groupInner_=8
> + reduceOutBuf 8V+预填 + pair-merge 形式), **未经 probe 验证**;
> **安装 .o = 4-chunk-no-events 构建(已知 L=512 fault)** —— 树与 .o 不一致!
> **恢复序**: ①网络恢复后先 probe(当前 .o 预期 fault, 确认即可) ②git checkout
> service_vector.h → 3dc61817(交付态) ③fast_op_rebuild ④probe 全绿 → 交付态恢复。
> **Cycle-X2 待验证命题**: groupInner_=8 + 8V + 预填 + pair 形式是否 fault
> (隔离 groupInner 8 的独立影响 —— c4e 曾证明 groupInner 8 + 8V + 4-chunk
> 过 512/1024, 但 pair 形式未测)。
> **本地留存**: patch_3way.py/.sh(3-chunk 缓存完整实现, 待上传应用)。

> ## 2026-09-30 下午终审: fold 复试确认回归, 已回退(3dc61817)
> 干净单发协议下的 fold 复试出现**65536 早触顶退化生成**(开头即有条目打满
> 65536 输出上限, 正常仅尾段 3-4 条) —— 与哨兵(max 1 闭合)互补的第二证据。
> **fold 终审: bf16 q_eff 舍入 = 真实 OOD 回归, 路线死刑**(OpDef 三输入同
> dtype, fp32 q_eff 不可行)。fixpipe 272µs 的正解回到 **cube 侧两级 GEMM
> 融合归约(fp32 精确)** —— 深度 kernel 工程, 设计已存本文件。
> **当前交付态**: e32e0d7b 等价(57ffed3a + pipe + pair-merge v2 + groupInner 8
> + ID2 同步), coarse@64K 357µs, gsm8k 97.65/97.88, AIME 76.67×2+63.33(噪声带)。

> ## 业界对标检索结果(2026-09-30, 网络搜索)
> 1. **[FlashAttention 反向算子 AscendC 优化案例](https://www.hiascend.com/developer/techArticles/20240607-1)**
>    (昇腾官网, 典型场景 **4 倍提升**): 优化手段 = Tiling 基本块调整 + 核间负载
>    均衡 + **CV 流水并行(Cube/Vector 解耦双流水)** + **MTE2 流水** + **FixPipe
>    流水** —— 与我们 msprof 三大瓶颈(fixpipe 63%/AIV vec 80%/标量等待 86%)
>    一一对应, 且证明同结构 4× 可达(与 LI 实测差距一致)。
>    同文镜像: CSDN 66693cc25eb5ae7f641ec684 / 掘金 7379147394755018792;
>    配套分析工具: CAModel + Profiling 流水占比定位。
> 2. **官方 SIMD 搬运指南**(hiascend 性能优化章节): 单次搬运块越大带宽利用率
>    越高(优先级高) —— 我们 fixpipe 实测仅 28.7GB/s 的病因之一。
> 3. **[TileRT](https://github.com/Tile-AI)**(Tile-AI 社区): tile 级常驻运行时引擎,
>    面向超低延迟 LLM —— **目标是 NVIDIA GPU, 非昇腾**; 其思想(算子分解为
>    细粒度 tile 常驻调度)与我们的 chunk 流水重叠设计同源; 昇腾侧对应生态 =
>    [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend) 与 AscendNPU IR。
>    (如需在此方向深挖, 需用户提供 tieRT 的确切全称/论文链接。)
>
> **可落地结论**: ①此前设计的"两级 GEMM 融合归约"(消 fixpipe 32MB 往返)
> 与 FA 案例的 FixPipe 流水/CV 解耦同路数, FA 案例证明该路线 4× 可达;
> ②实施时按 FA 案例配套用 CAModel/Profiling 逐流水占比验证每步收益;
> ③搬运块 ≥16KB 原则已满足(单次 CopyIn 16KB), fixpipe 小块写出是反例,
> 需靠 M=1 归约从源头消量。

> ## 2026-09-30 晨间交付(通夜闭环终态)
> **交付构建**: HEAD `4b92f6d9`(= 57ffed3a 同步粒度对齐 + pipe + fold 回退),
> 安装包已重挂, 服务运行中(健康), **GATE PASS**(确定性+召回全绿)。
> **通夜性能净收益**: coarse@64K/R4 443→**412µs(−7%)**(pipe 补丁); 更深的
> fold(−18%)与 pair-merge(−12%)均因**整网 AIME 证据链不确定**被回退保守处置
> —— 两者代码都在 git 历史(fold=被否; pair-merge=315d3333, cherry-pick 可恢复),
> AIME 噪声实验(同 kernel 4 跑 63.3-76.7)说明用 gsm8k(n=1319, 带宽 ±0.4)+AIME
> 多样本(≥3, 单发协议)联合判定后即可解锁, 解封路径明确。
> **4-chunk 缓存**: 三轮修复+二分后仍 fault@L=2048, 已回退; 归因=UB 零余量 +
> 未定位的 4 路分支残留问题(printf 插桩版已建未跑, 见 TODO)。
> **UB 审计**: 完整缓冲表+重预算方案(groupInner_ 16→8 腾 32KB, DoScale/
> DoReduce 语义已核实安全)已入本文件, 为解封后首实施项。
> **调试基建**: ccec_g 内核调试符号已接入(const_var.py×2, 未提交 —— 若确认
> 不影响性能可保留; addr2line 路径 = cann-9.1.0/tools/hcc/bin/aarch64-target-
> linux-gnu-addr2line)。
> **给上午的复核清单**: ①服务健康+构建一致 ②gate 快验(命令在本文件) ③
> AIME/gsm8k 历史分数见 gsm8k_ship.log/aime_final.log ④未竟事项=4-chunk 插
> 桩版(build_c4i2 分支态, printf 标记已置)+PD 网格任务(用户暂缓)。

> ## 用户指令的目标架构(2026-09-30): 单 fused pivot 算子 + 元数据前置
> **profiler 实证**(单 decode step, 用户采集): coarse 82.4 + refine 61.3 =
> 143.7µs, 其余全部算子合计仅 ~110µs → pivot 占整步 57%, 是唯一主目标。
> 小算子清单: aclnnMean×2(24µs, q̄/w̄) + Subs(1.3, seq-g) + Clamp(1.5,
> floor 1) + Contiguous(1.9) + int32 cast 若干 ≈ 30-40µs 可吸收。
>
> **设计三件套(全部 bit-identical 可达)**:
> 1. **q̄/w̄ 融合进 coarse**: 库内先例已存在 —— 本文件 M1-proxy 注释即
>    prefill 路径的 "AIV 全局预阶段写 proxy + SyncAll + CrossCoreWaitFlag
>    放行 AIC" 模式; decode 复用同模式: AIV 预阶段按 torch 同序(fp32 累加
>    → bf16 一次舍入)算 q̄/w̄ 写 GM scratch, AIC 经 CrossCoreWaitFlag 消费
>    —— 值与 python mean 逐位一致 → bit-identical ✓(与 fold 的差异: fold
>    改了数学(折叠), 本方案只搬位置不改值)。
> 2. **Sub/Clamp 融合进 coarse**: kernel 内收 seq_lens 原值 + 标量 g,
>    内部算 aslk = max(seq_lens-g, 1) —— 同算术, bit-identical ✓, 删 2 个
>    小 op; aslq 累计已在 metadata(tndMetadata 前缀和), 无需每步算。
> 3. **tile 流水**: AIC(fixpipe→cube)与 AIV(CopyIn→sort/merge)按 chunk
>    双缓冲深度重叠(mm1Res 段 DOUBLE_BUFFER 已有, 需加深 + AIC 标量事件
>    粒度细化); refine 同理评估其 CopyIn/sort/merge 三级。
> **tieRT 参考**: 未在库内找到该名(待用户给出全称/位置); 以本库 prefill
>    M1-proxy 模式 + LI 生产实现为参考基准。
>
> **实施顺序**: ②③(groupInner_ 8 + 4-chunk, 构建中) → 本三件套 → msprof
> 逐项对标 profiler 表 → 整网行为门。

> ## ①核实完成(2026-09-30 凌晨): DoScale/DoReduce 跨 GROUP 数据流已确认
> DoScale 尾部(vector.h:121-123): `outerGidx != 0` 时
> `Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner×s2Inner)` ——
> 后续 GROUP 的缩放结果**累加**进 reduceCacheBuf; DoReduce 对累加缓冲做
> 二分求和(rNum=groupInner_, pow2 收缩精确)。**groupInner_ 16→8 数学安全**
> (outerG 32/8=4 次迭代, 每次累加 8 行, DoReduce 二分对 8 行精确)。
> 实施细节注意: tmpBuf_ 内部布局(dbTmpUb[pingpong] + weightsInUb 偏移
> procGnum×s2BaseSize_)需按 groupInner_=8 逐字节重算核对(CopyIn 的
> mmUbStride 写法)。

> ## UB 完整审计(2026-09-30, InitBuffers 全量枚举, service_vector.h:135-176)
> over-2K 路径(sparseCount=4096, groupInner_=16, s1BaseSize_=4, V=512):
>
> | 缓冲 | 尺寸公式 | 实际 | 备注 |
> |---|---|---|---|
> | outQueue_ | max(16KB, offsets+16×512×4) | **~33KB** | reduceCacheBuf 需 32KB+ |
> | tmpBuf_ | (16×512+512)×2×4B | **68KB** | GROUP pingpong db, 最大消费者 |
> | sortOutBuf_ | 2×4096×2×4B | **64KB** | accumulator(globalTopkUb_) |
> | indexBuf_ | 512×4B | 2KB | |
> | reduceOutBuf_ | 512×5×4B | 10KB | 含 3 段保留裕量 |
> | brcBuf_ | 16×8×4B | 0.5KB | |
> | **小计** | | **≈177.5KB** | **winOutBuf_/winAuxBuf_(窗口阶段)不在本 Init, 待定位** |
>
> **关键发现 1(原代码即静默越界)**: reduceCacheBuf 需 32KB+ 而 outQueue_ 按
> `max(16KB, 32KB+)` 分配 —— 实际是按 32KB+ 分配的 ✓(我此前误读为 16KB);
> 即 **outQueue_ 无冗余, outNeedBufSize=reduceCacheSize≈33KB**。
> **关键发现 2**: UB 总量 ≈177.5KB + winOutBuf_(outW=4103×4B≈16KB, 待定位
> Init 处) ≈ **193.5KB ≈ 192KB 上限** —— 已零余量, pair-merge v1/4-chunk
> 的故障全部由此解释。
>
> **UB 重预算方案**(释放 32KB, 全部 bit-identical):
> `groupInner_ 16→8`(InitParams:193 "按 UB 空间强制为 16" 的注释):
> - tmpBuf_ 68→36KB(**腾 32KB**), outerG 2→4 次(单次 CopyIn 16KB 仍达
>   16KB/次搬运效率线 ✓), DoScale/DoReduce 逐 GROUP 逻辑不变;
> - **前置核实**: DoReduce(gRedCnt=min(groupInner_,gSize_)) 在 outerG 循环
>   后仅调一次 —— groupInner_ 语义与 32 头的覆盖关系必须先读 DoReduce 本体
>   确认(累加式则安全, 覆盖式则需循环内调用);
> - 释放后: reduceOutBuf_ 5V→8V(+6KB) 容 4-chunk 缓存段, 余 ~26KB 裕量。
>
> **实施顺序**: ①读 DoReduce 本体定语义 → ②groupInner_ 16→8 + tmpBuf_ 收缩
> → ③reduceOutBuf_ 8V + 4-chunk 缓存(补丁已在 git 历史/本报告) → ④probe→
> gate→bench→msprof → ⑤整网行为门(哨兵+gsm8k+AIME)。

> ## 4-chunk 缓存尝试失败归因(2026-09-30 晨, 两轮 VEC 越界, 已回退)
> 修复历程: ①首填垃圾读(缺 -inf/-1 预填) ②双向事件同步缺失 ③4 路输出
> 2048 对拆两次 1024-src acc 归并(MrgSort4 单队列上限实测 1024) —— 全修后
> **仍 fault@L=2048(4 路分支首触)**。Cycle A 二分已证明新增机制本身
> (Init/DataCopy/状态)即 fault, 与归并调用无关。
> **根因定性: UB 布局零余量纸牌屋**。实证: 原代码 reduceCacheBuf 需 32KB
> (groupInner_=16 × 512 × 4B) 而 outQueue_ 仅 16KB —— **原代码本就在静默
> 越界使用相邻 UB**(靠未用 padding 才没炸)。任何 +N KB 的新缓冲都会推垮
> 相邻缓冲。**解封前置条件 = 完整 UB 审计 + 重预算**(tmpBuf_ 68KB 的
> groupInner_=16 语义核实是最大潜在释放点), 非 patch 级可完成。
> pair-merge v2(315d3333, 已验证 −12%)不依赖新 UB, 是当前 UB 布局下的终点。

> ## 2026-09-30 08:00 交付终态
> **最终构建 HEAD = e32e0d7b** = 57ffed3a(同步粒度对齐 LI) + pair-merge v2(零新增 UB)。
> **性能**: coarse@64K/R4 **443→365µs(−18%)**; @256K 1581→1356µs(−14%)。
> **精度三重门全绿**: ①校准门全档 PASS(64K: PIVOT 0.61-0.66 vs LI 0.43-0.47)
> ②gsm8k n=1319 = **97.65**(K 带 97.04-97.88 内) ③哨兵 3 轮 eval 全部
> max 1 闭合零循环。
> **AIME 噪声终识**(5 个干净样本): K={76.67×3, 63.33}, V={70, 60, 76.67}
> —— 同 kernel 也能 63.3↔76.7 大幅摆动, 单跑 AIME 无判据; 大样本 gsm8k 才算数。
> **vs LI 剩余差距**(412+88 vs 119 → 现约 365+88=453 vs 119): AIV 排序机器
> 计算量(vec 80%)+fixpipe, 需 4-chunk 缓存(UB 重规划后)+两级 GEMM 融合归约
> —— 设计与本文件全部分析已就绪, 为下一轮工程迭代起点。

> ## 2026-09-30 晨间终判(pair-merge v2 洗清 + 保留)
> **双发污染定案**: 昨夜 v2 的 AIME 样本(66.67/70.00)与 45.56 gsm8k 全部是
> **双发假象**(两次 ais_bench 并发, 预测互相污染; 日志 `Launch TasksMonitor`
> 计数=2 实锤)。教训已入记忆: 重发前 pkill+复核 0+TasksMonitor 计数=1。
> **干净复测(v2r = 57ffed3a + pipe + pair-merge v2, HEAD e32e0d7b)**:
> - gsm8k n=1319: **97.65**(K 带 97.04/97.35/97.88 内) → 精度无损失
> - 哨兵: AIME 30 条 + gsm8k 1319 条全部 max 1 闭合 → 无行为回归
> - 校准门: 全 PASS(含 64K: PIVOT 0.61-0.66 vs LI 0.43-0.47)
> - 性能: coarse@64K/R4 **443→365µs(−18% 累计)**, @256K 1581→1356µs
> **终态**: 保留 pair-merge v2; AIME 单跑噪声带实测 63.3-76.7(同 kernel),
> 单跑 AIME 不能作为 ±10 分级回归的判据, gsm8k(n=1319)+哨兵才是判据。

> ## 下一枪设计: 4-chunk 缓存归并(LI 自己的 SortedBasicBlock_ 方案, 2026-09-30 凌晨)
> 现状: pair-merge v2(315d3333/e32e0d7b) 后 coarse@64K=365µs, 目标 coarse+refine
> < LI(119) → 需 coarse <~40µs, 还差 ~9×。
> **LI 的答案就在它源码里**(lightning_indexer_service_vector.h:382-395):
> actS1Size≤4 且非 over-2K 时, LI **每 chunk 只 Sort 不归并**, 排序结果存
> `SortedBasicBlock_[innerS1Idx×BASE_TOPK×2 + cacheIdx×s2BaseSize×2]`
> (4×512×2×4B = **16KB/行**), **每 4 个 chunk 才 MrgBasicBlock 一次**进
> globalTopkUb_ —— 归并频率是我们的 1/4。
> **UB 重预算**(现 ~176KB/192KB): ①tmpBuf_ 68KB 中 groupInner_=16 是
> "按 UB 空间强制为 16"的注释 —— fold 后 mm1 输出仅 [1,512], groupInner_
> 可降到 4 → tmpBuf_ 68→20KB, **腾 48KB**; ②新增 SortedBasicBlock_
> 4×512×2×4=16KB/行×4 行=64KB?? 超 —— 按 innerS1Idx≤2 分配或缓存 2 chunk
> (pair 的 4-chunk 版)。**精确 UB 表在实施前必须逐项列**。
> 实施前提: fold 已回退, groupInner_=16 是 H=32 口径所需(每 GROUP 处理
> 16 头? 需核实 groupInner_ 语义后再动)。
> **预期**: 归并频率 1/4 + pair-merge 叠加 → coarse ~200-250µs; 不足以达标,
> 达标还需 fixpipe 272µs 的治理(两级 GEMM 融合归约, 设计已在 §Cube 侧两级)。

> ## pair-merge Cycle A 判决 + Cycle B 设计(通夜迭代, 2026-09-29 晚)
> **Cycle A(bisect)判决**: 归并已回原版仅保留新增机制仍 fault → 死因 =
> 新增 32KB pairAcc 的 UB 预算(UB 实际余量 = 0, 窗口缓冲 winOutBuf/winAux
> 未计入旧估算)。**UB 已确认无任何余量, 任何新 UB 缓冲的方案直接出局**。
> 已修复的真实 bug(留给正式版): ①首填 DataCopy 缺双向事件同步
> (V_MTE2 源就绪 + MTE2_V 拷回完成) ②InitSortOutBuf 计数单位=float 非 pair
> ③InitSortOutBuf 尾半槽必须预填(-inf/-1), 否则 1024 宽归并读垃圾。
>
> **Cycle B 设计(pair 态下沉 workspace GM)**:
> - 位置: mm1 段尾与 candidatesWs 段之间插 32KB(=4 行 × 1024 对 × 8B);
>   kernel.h 的 candidatesWs 绑定点 = mm1Size 锚点, host workspaceSize 同步
>   +32KB —— 两处单一锚点, 无 reqNum 依赖;
> - kernel: pairAcc 用 `candidatesWsGm_[偏移].ReinterpretCast<float>()`,
>   DataCopy(UB→GM) + 下 chunk DataCopy(GM→UB), 事件对同已验证模式;
> - 每对代价 +2 次 4KB GM 小拷贝(~1-2µs/2chunk), 净收益保留 ~15-20%;
> - 迭代方式(用户确认): 已验证 commit 57ffed3a 不动, 迭代在树上进行,
>   不必回退; 重建不影响已运行服务(用户确认); NPU 被占则轮询等空闲。

> ## LI 逐管道对照(msprof op 同法采集, R=4 L=64K, coarse 433.5 vs LI 119.4µs)
> | 管道 | coarse | LI | 差距 |
> |---|---|---|---|
> | AIC cube | 48.5 | 26.0 | LI 量大 4×(M=128 vs 32)仍快 2× |
> | AIC mte2 | 198.9 | 49.4 | 4.0× |
> | AIC fixpipe | 272.2 | 73.0 | 3.7×(LI 量大 4×) |
> | AIC mte1 | 93.2 | 20.9 | 4.5× |
> | AIC scalar wait_id0 | 361.4 | 84.2 | 4.3× |
> | AIV vec | 346.9 | 84.0 | 4.1×(LI 排序行数 4×) |
> | AIV mte2 | 52.8 | 52.1 | 1.0×(LI 量 4× 带宽用满, 我方延迟受限) |
>
> **结论**: 结构完全相同(S2_BASE_SIZE=512、mm1Res GM 往返、DoScale/DoReduce、
> 同款 SortAll/MergeSort), LI 每根管道 uniformly ~4× 高效 —— 微观执行差异
> (缓冲编排/流水重叠/指令编排), 非架构差异。优化 = 逐管道以 LI 为对标,
> 源码: `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/
> ai_core/tbe/custom_transformer_impl/ascendc/lightning_indexer/arch22/`
> (kernel.h / service_cube.h / service_vector.h / vector.h 全套可读)。
> 复现采集: probe_prof.py(coarse) / probe_prof_li.py(LI), msprof op 方法
> (目标算子 = 脚本首个 aicore op)。

> ## pair-merge 尝试失败归因(2026-09-29, 两轮 aicore VEC 越界, 已回退)
> 设计本身成立(top-k 关联律 bit-identical), **死因 = UB 预算**: 现 UB ~160KB
> (sortOutBuf 64 + tmpBuf 32 + reduceOutBuf 40 + outQueue 16 + index 8),
> pairAcc 每行槽 8KB × cuS1ProcNumPerAiv(4) = 32KB → 192KB 顶满越界
> (8 槽 64KB / 4 槽 32KB 两轮均 fftsplus aivector 越界, fixedPC=VEC)。
> **解封条件**: 先做 UB 重规划 —— reduceOutBuf "5 段保留裕量"可裁至 4 段
> (省 8KB), tmpBuf_/outQueue_ 生命周期分析后或可再挤; 或 pair 态下沉 GM
> (candidatesWsGm 有富余, 加一次 GM 往返 ~2KB/chunk, 仍净赚)。
> 交接状态: service_vector.h 已 git checkout 回 57ffed3a, 重建恢复中;
> 补丁全文在本文件 git 历史与 patch_pair.sh 思路(5 处锚点已验证可落)。

> ## 下一枪设计: chunk 对合并(57ffed3a 之后, 预期再省 AIV 归并 ~30%)
> 现状(57ffed3a): 每 512-key chunk = SortAll(512) + MergeSort(chunk, acc4096)
> + 32KB 拷回。**pair-merge**: 两 chunk 先 MergeSort(chunkA, chunkB→pairAcc
> 1024), 再一次 MergeSort(pairAcc, acc4096)。top-k 关联律保证输出
> bit-identical; 每对省一次 4096 宽归并 + 一次 32KB 拷回。
> 实施要点(service_vector.h S2 循环):
> - 新增 InitBuffer pairAccUb_(virTopK_pair=1024 ×2 ×4B = 8KB, InitBuffers);
> - per-innerS1Idx 的 pairFilled 标志(数组, 与 blockS2StartIdx_/S1 边界
>   同步重置); 奇数尾 chunk 在 isS2End 时 flush;
> - MergeSort 调用序: pairFilled ? [MergeSort(pairAcc,1024,chunk);
>   MergeSort(acc,virTopK,pairAcc); pairFilled=false] : [MergeSort(pairAcc,
>   1024,chunk); pairFilled=true]; needCopyOutGm 前保证已 flush;
> - 验收: gate(形状+确定性+召回逐位==前构建) + bench + msprof 对比 +
>   整网行为门(</think> 哨兵+AIME≥76)。
> 风险: UB 预算 +8KB(当前 ~160/192KB); S1/块边界状态重置遗漏 → 确定性
> 压测暴露。

> ## msprof op 基线分解(2026-09-29, R=4 L=64K, 433.5µs, mix 24核)
> 采集法: `msprof op python3 probe_prof.py` —— **脚本必须让 coarse 成为首个
> aicore 算子**(全 CPU 构造 + .npu() 搬运; msprof op 单算子模式只抓第一个)。
>
> | 通道 | 耗时 | 占比 | 结论 |
> |---|---|---|---|
> | AIC fixpipe(L0C→GM 存 mm1Res) | 272µs | 63% | H=32 输出宽, 每 chunk Fixp 串行存储 |
> | AIC scalar | 324µs | 75% | 其中 **wait_id0=361µs** —— 86% 在等 L0 缓冲事件, 串行化铁证 |
> | AIC mte2(GM→L1 key) | 199µs | 46% | 与上述重叠 |
> | AIC cube(真打分) | 48.5µs | 11% | 打分本身极便宜 |
> | AIV vec(sort/merge/掩码) | 347µs | 80% | 排序机器真计算量, fold 后即瓶颈地板 |
>
> ### 优化优先级(按此画像重排)
> 1. **AIC L0 缓冲数**(wait_id0 361µs 的直接对症): L0_BUF_NUM/l0BufIdx 加深
>    或 Fixp 批量化(mm1Res 双缓冲累积后一次 fixpipe), 消 86% 标量等待;
> 2. fold 重试的前提解: fold 省的正是 fixpipe 272µs 的大头(M=1 → 存储宽 /32)
>    —— 若 fold 行为回归能通过"fold 后整网微调"或"训练侧同构"解决, 长期最优;
> 3. AIV 347µs 排序地板: MergeSort 事件握手 + 拷回 32KB 免除(设计已存);
> 4. LI 逐行对比(源码在 vendors/custom_transformer/.../lightning_indexer/arch22/)。
> 全部改动的验收: gate(形状+确定性) + msprof 对比 + 整网行为门(</think> 哨兵)。

> ## Phase 1 排查报告(ascendc-operator-performance-optim 规范, 2026-09-29)
> 基准备份: `csrc/attention/indexer_coarse_screen/op_kernel_bak_20260929`(原工程未动)
>
> ### 发现的问题(按预期收益排序)
> 1. [阶段5 流水] 每 chunk 两次全序化: 预排序 `PipeBarrier<PIPE_ALL>`(已降
>    PIPE_V, 实测仅 +1-2%, 说明非主因) + **MergeSort 尾部 PIPE_ALL**
>    (vector.h:294-298, 128 次@64K, MTE2 拷回 vs 下块 MrgSort 读的竞争修复)
>    → 应降 MTE2_V 事件握手(入口 WaitFlag+pending 守卫, InitSortOutBuf 前
>    补 Wait; MIX 核事件空间与 cube 共享, EVENT_ID0/1/2/3/5 已占, 待审 4/6/7)
> 2. [阶段2 搬运] MergeSort 拷回 32KB/chunk UB→UB DataCopy(4096×2×4B),
>    128 chunk = 4MB 纯搬运; MrgSort dst 写-only 尾部约束限制直写,
>    需查 MrgSort 双缓冲免拷形态
> 3. [阶段5 流水] chunk 间无重叠: CopyIn(N+1) 与 SortAll/MergeSort(N) 串行;
>    pingpong 仅用于 GROUP 维(outerGidx), chunk 维单缓冲 → 2.6-3.4µs/chunk
>    全串行, LI 同结构 0.93µs(其内 32× 打分并行完成)
> 4. [阶段1 Tiling] L1 已满(KEY_BUF_NUM=3×128KB + query 2×64KB = 448/512KB)
>    → chunk 无法加大(2048 实测 AICORE 崩, 已回退); 若 KEY_BUF 3→2 腾
>    128KB 可试 s2BaseSize=768(chunks 128→85), 但牺牲一级预取流水
>
> ### 已确认无问题/已排除
> - 窗口注入(hw0/1: 1µs)、gather 模式(sorted 1.00-1.04×)、打分计算量
>   (fold M=1 后 1/32 仍慢 2.8×)、排序算法本体(逐字节复刻 LI)、
>   fp32/BF16 反模式(打分已 fp32 累积)
>
> ### 优化计划(本轮目标)
> 按收益: ①MergeSort MTE2_V 事件握手 → ②chunk 双缓冲重叠 → ③KEY_BUF/768。
> **硬约束(fold 教训)**: 输出必须 bit-identical(不改模型所见), 每步过
> gate(形状矩阵+确定性压测) + 整网行为门(</think> 计数哨兵 + AIME 分数)。
> 执行流程: 服务释放后 msprof op 基线(OpBasicInfo Task Duration + Pipe
> Utilization) → 备份目录内改 → 精度门 → 对比报告 → 原���程替换 + commit。

> ## 2026-09-29 第二轮(fold 落地 + 流水补丁)
>
> ### 已落地(commit b6908a07 + 流水补丁待计时)
> | 项 | 结果 |
> |---|---|
> | fold(python 层) | coarse@64K/R4 **447→334µs**; 超集召回 0.576→**0.999**@64K |
> | 校准门(端到端) | **PIVOT 反超原生 LI**: +0.02@8K, **+0.06@16K**, 全 MTP3 行 ok, 确定性压测 ok |
> | 流水补丁 #1 | 每 chunk -inf 预填(整块时)省去 + 排序前 PIPE_ALL→PIPE_V; 已重建, gate PASS(待计时) |
> | 整网验证 | fold build 服务重启 + AIME 2024 运行中(65536 out); gsm8k 排队命令有 bug 已弃用 |
>
> ### fold 的精度机理(意外收获)
> H=32 口径的 0.576 召回损失 = 32 路 bf16 逐头点积在 top-k 边界的噪声;
> fold 后单次 K=128 点积(fp32 累积)几乎无噪 → coarse 超集近乎 oracle 精确。
> **fold 同时是精度修复而非性能取舍** —— 这改变了 PIVOT 的设计叙事:
> 长 L 下 PIVOT 候选质量显著优于原生 LI 的 top-k。
>
> ### 已排除的路(勿重试)
> - device 侧 s2BaseSize 放大(512→2048): L1 三缓冲 key tile 已满
>   (KEY_BUF_NUM=3, 448/512KB@512), chunk 无法加大, 已回退;
> - host workspace 跟随放大: 与 kernel 不一致会 EZ1008(mBaseSizeAlign 对齐
>   128 使每 buffer=1MB), 若未来重试必须两处同步;
> - 有序 vs 乱序候选池: 1.00-1.04×, gather 模式非瓶颈;
> - gsm8k 排队健康检查: 不能写 127.0.0.1(服务在 .82, 评测在 .218)。
>
> ### 下一轮( MergeSort 事件握手, 设计已定待实施)
> MergeSort 尾部(vector.h:294-298)每 chunk 一次 `PipeBarrier<PIPE_ALL>`
> (2026-08-30 数据竞争修复: MTE2 拷回 vs 下块 MrgSort 读) —— 可降为
> MTE2_V 事件握手: 出口 SetFlag(MERGE_EVT) + 入口/InitSortOutBuf 前
> WaitFlag(pending 守卫)。**风险点**: MIX 核事件空间与 cube 共享
> (EVENT_ID2/5 已用), 需全量审计分配; InitSortOutBuf(V) 与悬挂 MTE2 拷回
> 的竞态必须覆盖。预期收益: 每 chunk 再省 ~0.5-1µs(128 chunk@64K)。
> 定量分解工具: msprof 指令级 profiling 定位 2.6µs/chunk 的构成。

> 2026-09-29 / 基准数据: 本目录 `ctx_sweep.csv`、`refine_pool_sweep.csv`(入图口径)
> 复现: `python bench_two_ops_ctx_sweep.py --csv ctx_sweep.csv`(NPU 空闲时)
> 配置: 910_9391, GLOBAL+L3+PIVOT=1 构建(7d691f40 后), 受控全行同长 aslk=L
>
> ## 2026-09-29 下午更新(目标①闭环 + ②③实施交底)
>
> ### A. 计时口径升级(响应"要看入图平稳耗时")
> bench_wall 已改为 **graph capture 口径**: warmup 后 K=10 次调用捕获进
> NPU graph, replay 计时/K —— 零 host 下发开销, 等价生产 FULL_DECODE 回放态。
> 复测结论: coarse 几乎不变(设备侧瓶颈实锤); LI 小 L 档降 ~10-15%(host
> 开销被剔除)。本目录 CSV/PNG 均为新口径。
>
> ### B. 精度门已建成并 PASS(目标①, `precision_gate_full.py`)
> 四段: 形状矩阵(R{1,3,4,16,32}×L{1..65536} 含非对齐/块界/chunk 界) +
> 端到端校准门 + 确定性压测(50 连发无 host sync、4 负载轮换、逐位比对
> —— ProcessWindow 跨核竞态类缺陷的函数级暴露)。
>
> **方法论要点(踩坑记录, 下游必读)**:
> 1. **绝对召回阈值不可用**: bf16 打分噪声下, 原生 LI 自己对 fp32 oracle
>    的召回只有 0.73@8K/0.66@16K/0.55@64K(选择比越低越低)。门必须校准:
>    PIVOT 链最终 top-2048 对 fp32 oracle 召回 ≥ 原生 LI 同输入召回 − 0.02。
> 2. **MTP 行相关性决定 mean-proxy 成败**: 4 行完全独立随机是 mean-proxy
>    的最坏情况(coarse 包含率仅 0.26@16K); 相关行(行0 + 0.1ε 模拟真实
>    draft)下 PIVOT≈LI(差≤0.001)。独立行数字不代表生产, 勿据此下结论。
> 3. **refine 数值上几乎完美**: 损失分解(L=16K)显示 op 实际召回 ≈ 完美
>    refine 上限(差 0.01-0.07), 全部损失来自 coarse 包含率的打分噪声 +
>    设计近似 —— 优化只动 coarse 不会伤 refine。
> 4. fold 恒等已实证: Σ_h w_h(q_h·k) 与 (Σ_h w_h q_h)·k 两参考的 top-k
>    完全一致(0.8462=0.8462)。
>
> ### C. 目标②③实施交底(锚点已核, 待落地)
> 设计: 权重收缩+mean 融合, mm1 从 (M=32头,K=128) 变 (M=1,K=128):
> - python: `pivot_indexer.py:113-114` 两个 mean 换成一个
>   `q_eff = einsum('rhd,rh->rd', q_bar.f32, w_bar.f32).bf16`
>   (mean 开销随之消失 —— 目标③);
> - op_host/tiling: actMBaseSize 按 M=1/请求 计算(原 s1×g 展平 32);
> - cube `ComputeMm1`(service_cube.h:143): M 维 32× 缩小, QueryNd2Nz/
>   LoadQueryToL0a/ComuteL0c/Fixp 全部随 M 缩; mm1ResGm 流量 32× 缩;
> - vector(service_vector.h:271-360): 删 DoScale/DoReduce/brcBuf 链
>   (~80 行), CopyIn 直接收每 key 标量; SortAll+MergeSort 不动;
> - torch_adpt: q_bar 形状 [R,1,128] 或保留 [R,32,128]+tiling 折叠标志
>   (二选一, 主力 agent 定)。
> 验收: 精度门 PASS(bench 域) + 整网 gsm8k/AIME 复测 +
> coarse@64K/R4 ≤ 60µs(现 443), chain/li < 1(长上下文挡)。
> 理论极限: coarse 每行工作 = LI 的 1/4(M=1 vs 4), 若达到 LI 的每 key
> 效率, coarse@64K/R4 ≈ 115.6/4 ≈ 29µs —— 即"理论省 4×"。

---

## 1. 基准事实(全 24 格 ok,refine 池宽恒 4103 生产口径)

| 维度 | 实测 |
|---|---|
| coarse | ∝ 单行 L,与 R≤16 无关(R32 翻倍=核数饱和): 64K→448µs, 128K→865, 256K→1700(R≤16) |
| refine | 与 L 无关、∝ 4R×W 线性: req4 恒 ~90µs(req4 W2048→8192: 62→121µs) |
| 原生 LI | ∝ 4R×L 全扫描: @64K req4→126µs, req32→632µs |
| chain/li | req4@256K 最差 **4.51×**; req32@256K 最好 **1.56×** —— 全网格无一处 chain 优于 LI |

**每 key 每行效率**: coarse 1.6ns(key·row) vs LI 0.085ns(key·row) = **19× 劣化**
(斜率分解: coarse @R4 43µs@2K→161µs@20K = 6.4ns/key/4row;LI 38→63µs = 1.36ns/key/16row)。

已排除因素:
- **窗口机制**: hw=0 与 hw=1 在 64K 处 449.4 vs 448.3µs —— 窗口/去重开销可忽略;
- **launch 固定开销**: 2K→20K 线性外推截距 ≈ 30µs,长序列区纯扫描主导;
- **带宽墙**: coarse @64K/R4 有效带宽 150GB/s(HBM ~5%),LI 2.1TB/s —— coarse 是
  计算/访存模式劣,不是带宽到顶。

## 2. 热点定位(kernel 代码链路)

`indexer_coarse_screen_service_vector.h:271-360` 每 S2 块:

```
Cube: mm1 = q_bar[h,:]·k[key,:]   (逐头, 输出 32头 × L × fp32 → GM)   ← 32× 放大
Vec : CopyIn(mm1ResGm ← GM 往返)  → DoScale(×w 广播, 逐头) → DoReduce(Σ32头)
      → 掩码 → SortAll + MergeSort(与原生 LI 生产同款)
```

即:**打分在 Cube 逐头算完后,32 头 × fp32 全量落 GM,Vector 再搬回来逐头乘权重、
逐 key 归约**。64K/R4 一遍 = 32×65536×4B×2(写+读) ≈ 33MB GM 往返,外加
DoScale 的 32 次 Mul repeat + DoReduce,全部逐 key 计价。

## 3. 优化设计(主案):权重前置收缩,mm1 直出每 key 标量

数学恒等(求和次序差异仅舍入级):

```
score[key] = Σ_h w[h]·(q_bar[h,:]·k[key,:])
           = (Σ_h w[h]·q_bar[h,:]) · k[key,:]  ≡  q_eff[128]·k[key,128]
```

改动三步:
1. **host/前置**(pivot_indexer.py 调 coarse 前,每请求一次,开销可忽略):
   `q_eff[r,d] = Σ_h w_bar[r,h]·q_bar[r,h,d]`(fp32 计算,bf16 落盘);
2. **Cube**: mm1 从 (M=32头, K=128) 变为 (M=1, K=128) —— cube 工作量与 GM
   产出**缩 32×**(fp32 每key 4B 标量);
3. **Vector**: 删 DoScale/DoReduce/brcBuf 整条链(约 80 行),CopyIn 直接收
   每 key 标量 → 掩码 → SortAll+MergeSort 不动。

预期收益: 向量尾与 GM 往返占当前每 key 成本的大头,若按 LI 的每 key 效率
折算,**448µs@64K → ~35-60µs**(14×);保守只吃一半也有 ~7×。

精度验证: fold 引入 bf16 一次舍入(q_eff 落盘),须过 precision_gate_two_ops
top-4096 召回门(现有 §5.5/5.6 方法论)。若 bf16 舍入不可接受,备选:
q_eff 用 fp16 落盘、或 cube 侧 K=4096 联合收缩(h,d 摊平进 K 维,精度同现路径)。

## 4. 量化目标(验收线,chain < LI)

| 场景 | 现状 chain | LI | 需 coarse ≤ | 现值 | 所需加速 |
|---|---|---|---|---|---|
| req4/64K | 540 | 126 | **34µs** | 448 | **13×** |
| req4/256K | 1792 | 397 | 300µs | 1695 | 5.6× |
| req32/64K | 1221 | 632 | 280µs | 869 | 3.1× |
| req32/256K | 3730 | 2398 | 2035µs | 3367 | 1.65× |

结论: 主案落地后若达 LI 级每 key 效率,**全网格 chain<LI 可达**
(refine 在 L≥64K 已稳定快于 LI,短 L 见 §5 旁路)。

## 5. 快赢(与主案正交,host 侧一行判)

**短上下文旁路**: L ≤ 4103 时候选池 ≥ 上下文,粗筛无裁剪收益纯属开销
(2K 挡 chain/li = 2.65~5.41)。`L ≤ 4103 → 直接调原生 LI`(同一 top-k 语义),
2K/10K 挡立即回本。

## 6. 次要观察

- refine: W=2048 时 req4 仅 62µs(比 4103 省 31%),若精度门证实 2048 预算
  召回足够,refine 侧再省一半(pool_sweep 曲线已给全);
- pool_sweep 曾误传 aslk=L 致 25× 虚高,已修正为 aslk=W(=每行有效候选数,
  生产 coarse 输出 aslk_op≈4103),修正后与 ctx_sweep 公共点对齐
  (90.1 vs 91.6µs)—— 复现时勿回退该口径。

## 7. 交付物(下游画图/复现)

- `ctx_sweep.csv` / `refine_pool_sweep.csv` —— 图数据(24/16 行,status 非 ok 自动剔除)
- `plot_ctx_sweep.py ctx_sweep.csv out.png` —— 复现大图(2×2 子图 × 三柱
  coarse/refine/LI),`ctx_sweep.png` 为当前渲染
- `bench_two_ops_ctx_sweep.py` / `bench_refine_pool_sweep.py` /
  `precision_gate_two_ops.py`(常量 shim, 已对照 pivot_indexer.py 核实)
