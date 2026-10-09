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
