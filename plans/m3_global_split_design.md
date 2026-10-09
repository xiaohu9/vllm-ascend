# M3 设计稿:SplitCore 全局块流均分 + 两级 topk 归并(2026-10-09)

## 问题(消融实证)
coarse 只切整行(行间并行、行内串行):R≤24 单行独占一核,512 块串行扫 → 单行延迟恒定
≈390µs(R4≈R16,R32 翻倍)= 台阶,违反 LI 式 R 线性度。
LI SplitCore(lightning_indexer_kernel.h:286):totalBlockNum(全局 行×s2块 流)/coreNum 均分,
每核一段连续全局块区间(bN2Start/gS1Start/s2Start..End)→ 时间 ∝ R 线性。

## 关键有利条件(已核实)
- 主 pass 循环已支持 s2 范围消费(splitCoreInfo.s2Start,首行后归 0)。
- V 侧为增量 pair-merge MrgSort 维护运行 top-4096(per row),行尾 CopyOut。
- 预阶段(M1a)行级单写者,与主 pass 切分解耦,不受影响。
- AIC/AIV 同 pair 同 range,乒乓对结构不变。

## 改动点
| # | 位置 | 改动 |
|---|------|------|
| 1 | kernel.h SplitCore | 移植 LI 式全局块流均分(输出含 s2 范围;空核语义保持:范围空=不参与同步) |
| 2 | workspace | 新增 partialTopkGm:每核 × 行片段 槽(globalTopkUb 4096×2 fp32=32KB);host tiling 增容量 |
| 3 | vector 主 pass 行尾 | needCopyOutGm 改写 partialTopkGm[coreIdx][frag] (DataCopy 32KB),记录本核行片段表 |
| 4 | 归并 phase(新) | 行 r 的末块归属核 = 行属主:读该行跨核连续区间 [cstart,cend] 的 partial 级联 MrgSort → 最终 top-4096 → CopyOut |
| 5 | ProcessWindow | 不变(属主=CopyOut 写者,同核 MTE3→MTE2 有序前提保持) |
| 6 | 空行/identity 路径 | 不变(无块行仍走 EmitIdentityRows) |

## 归并成本
R4:2048 块/24 核≈85 块每核,每行跨 ~6 核 → 归并 6×4096×8B=196KB 读 + 级联 MrgSort ≈ 15-30µs。

## 预期
R4/L64K:393 → ~130µs(主 pass ÷6 并行);R 严格线性(时间 ∝ 每核块数)。
M2(塌缩,plans/m2)随后叠加,砍每块 V 工作量(CopyIn ÷32,DoScale/DoReduce 消失)。

## 验证链
probe_m1a 7 档硬门(逐位;tie 由 -inf 掩码结构保证)→ precision_gate → bench 4 档
(重点看 R 线性度:R4/R16/R32 斜率)→ AIME。
## M3 实施规格定案(2026-10-09,全部锚点已读)

### 关键事实
- 主循环 (bN2,gS1,s2Idx) 三元组全程通用:s2LoopEnd=isEnd?s2End:s2BlockNum-1;V 侧 cuBaseS2Idx=s2Idx*256。
- over-2K 现状:SplitCore 的 s2Loop=1(每行 1 单元)+ s2End=s2BaseNum-1 三元 → 整行单核。
- AIC/V 乒乓协议(ProcessBaseBlock 每 Call 一对 flag)对子块化天然泛化。
- 双 AIV 按 innerS1 半区分工(blockId_%2),CopyOut 各拷各的半区 → partial 槽按 (pairIdx*2+seq) 布局,双 AIV 写同一 256KB 区的不同 innerS1 槽,无冲突。
- V 侧增量 topk(pair-merge MrgSort)在任意子区间内照常工作;fragEnd 时 flush deferred(pairFilled_)再写 partial。

### seq 规则(writer/owner 双端 O(1) 一致)
- fragment 分类:head(s2Idx>0 开始)→ seq=1;tail(含行首 s2Idx==0)→ seq=0。
- owner(行尾 fragment)= tail:uEnd = unitStart(self)+fragLen(s2End-s2Start+1);uStart = uEnd - s2BlockNum;
  cstart = max{c: unitStart(c) ≤ uStart}(≤24 次反向步进);对 c∈[cstart..cend],seq_c = (c==cstart)?0:1。
- 未切分行(≤cap 单元)单核独占 → cstart==cend==self,归并退化为空,路径统一。

### workspace
partialTopkGm = 48 槽(24 pair × 2 seq) × 256KB(=8 innerS1 × 4096×2 fp32)= 12MB;无目录。
tiling.cpp GetWorkspaceSize 增;kernel Init 绑定后经 InitVec1GlobalTensor 传入 service。

### 改动文件
1. kernel.h SplitCore:s2Loop=s2BaseNum(去三元/break-hack/isLD);ConstInfo 增 unitCap/deal1More/totalUnits(SplitCore 填);
   CalcRunInfo 的 RunInfo 增 unitStartSelf(=coreIdx×cap+min(coreIdx,deal1More))。
2. common.h:RunInfo + ConstInfo 字段。
3. tiling.cpp:workspace。
4. service_vector.h:成员 partialTopkGm_/fragIsHead_;ProcessVec:
   - isFirstS2InnerLoop: fragIsHead_=(info.s2Idx!=0)
   - fragEnd(=info.isLastS2InnerLoop): flush deferred → DataCopy partial[(pairIdx*2+fragIsHead_)]
   - isS2End(=owner): 写自身 partial 后,读 [cstart..cend]×对应 seq 的 partial(innerS1 限本 AIV 半区),逐个 MergeSort 折入 globalTopkUb_,走原 CopyOut 体
5. service InitVec1GlobalTensor +kernel.h 调用点:+partialTopkGm 形参。

### 验证
probe 7 档硬门(n_diff 必须 0)→ precision_gate → bench4(R4/R16/R32 斜率=线性度)→ AIME。

## M3 调试进展(B1-B5 二分,2026-10-09 深夜)
- B2 定案:挂死根因 = ProcessMain 入口 usedCoreNum(行基)早退杀掉已分到子块的 pair 的 AIC → 其 AIV 永等 C1V1。已删早退(空范围由 isEmptyRange 跳过;空 pair AIC drain 有 4 种子)→ R4/R32 不再挂死。
- B3 定案:SyncAll 放主循环尾 → 507015(撞乒乓尾巴);B4 移到 ProcessWindow SyncAll 后 → R4/R32 不挂但 det=False(竞态)。
- B5:MergeSort srcLen=8192 超段长 → 507015(L2K 复现);改 8×1024 级联 → R4/R32 无挂死。
- **当前唯一缺口:跨核 partial 可见性** —— det=False(非确定)= 文档记载"SyncAll 不保证跨核 GM 可见性"的 AIV 版;partial 在途即被属主读。
- **修复方案(下回合)**:marker-poll(本仓 M1a PreStageBarrier 已验证范式):fragment 写 partial 后同核 MTE3 FIFO 追写 magic 到 dirGm[core];属主对 [cstart..self) 逐核 spin-poll magic(UB DataCopy+GetValue 重试)到齐后才折叠。备选:mode0 flag 配平(计数器 0-15 会被 in-flight 集合冲爆,弃)。
- B5 现状数据:R4 n_diff=7724 det=False;R32 n_diff=4096 det=True;L2K(碎片更碎)507015 待 B5 后复测。
