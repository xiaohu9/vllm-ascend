# M2 设计稿:w̄ 折叠进 q̄(线性塌缩)+ mm1 直出分数(2026-10-09)

## 动机(消融实证,2026-10-09 NPU)
- coarse(mode1) 单核 405µs/发(R4/L64K),graph-replay 墙钟 393µs —— 纯 kernel-bound。
- L 主导:64K→16K 省 251µs(~5.2µs/千 token);R 无关(16 行 +4.5µs);window 通道仅 11µs。
- 每 key 成本 = mm1 逐头 32 FMA + V 侧 DoScale/DoReduce(H=32 归约)。

## 数学依据(双线性塌缩,恒等变形)
score[key] = Σ_h w̄[r,h]·(q̄[r,h]·k[key,:]) ≡ q̄'[r]·k[key,:]
其中 q̄'[r] = Σ_h w̄[r,h]·q̄[r,h,:] ∈ R^{Dh}。
M1a 预阶段已在核内算出 q̄/w̄ —— 顺势多算一次折叠,H=32 的逐头维度整个消失。

## 改动点(全部改动点清单)
| # | 位置 | 改动 | off/兼容 |
|---|------|------|---------|
| 1 | service_vector ProcessGroupMean | q̄ 累加循环内同步做加权:acc'[d] += w_row[h]·q_row[h,d];输出仅 q̄'[R,Dh] | mode0 不走预阶段,不受影响 |
| 2 | kernel workspace 绑定 | qBarWs 从 reqNum*H*DH 缩为 reqNum*DH(wBarWs 删除);tiling/host/kernel 三处同步 grep | tiling 字段增删 → 全量再生 |
| 3 | cube 侧 mm1 | q̄'[s1,Dh] × kᵀ → 直出 [s1,s2] fp32;删 H 维循环 | 主 pass 结构简化 |
| 4 | vector 侧主 pass | 删 DoScale/DoReduce/indexBuf 相关逐块工作 | 每块 V 工作≈0 |
| 5 | refine 不动 | refine 用 raw q_li,与 q̄ 无耦合 | 0 改动 |
| 6 | host 语义 | aslk/candidates 契约不变 | 0 改动 |

## 数值影响
- q̄' 多一次 bf16 舍入(预阶段 fp32 累加后单次舍入,与 M1a 同风格)。
- score 由 fp32 mm1 累加直出,较现链少一层 per-head 中间存储舍入(如现 MM1_OUT_T=bf16 则更准)。
- probe_m1a A/B 仍成立(两 mode 走同一新核);precision_gate 召回门(≥0.98)预期通过;AIME 为最终裁决(用户已批:aime 有问题再说)。

## 验证链(照 M1a 流程)
1. rebuild.sh + syncall(注册面变更 → rm -rf autogen impl binary 全量再生)
2. probe_m1a 7 档 ALL-PASS(硬门)
3. precision_gate_full.py 召回+确定性
4. bench_m1a 4 档 → 期望:365µs → ~150-200µs(消 V 侧+H 维,逼近 kv 16MB 带宽地板 ~80-100µs)
5. serve + AIME(流程见 memory/aime-gate-flow)

## 风险
- mm1 形状变化触发 tiling key/模板分支 —— 需 grep tiling_key 模板注册面。
- DoScale/DoReduce 删除后 indexBuf_/reduceOutBuf_/brcBuf_ 等缓冲大小可回收(留给 s2BaseSize 增大的二期)。
- graph_smoke S6b 的 pad 行为:q̄' 路径同样受 aslk floor=1 保护,语义不变。
