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

## 实施蓝图(2026-10-10 凌晨勘测,M3 收官后执行)

### 数据流事实(代码勘测,非推断)
- cube 侧已是 K=headDim(128) 单 GEMM:Mmad(k=128) → Fixp 直写 mm1ResGm [m, s2Align] fp32,双缓冲 loop%2,host 预留 512*512*4*2B/核 ✓。
- H 头在 **M 维**:query M 行 = 每请求 g(=4) 个组行(q̄[r,g,128] 摊平);权重 = 每请求 g=4 个(weightsCoreOffset = prefix*qHeadNum(=4=gSize) + n2Idx*gSize)。
- V 侧 DoScale:weights[groupInner] 广播乘 mmOut 组行;DoReduce:二分加组行 → 1 行 s2 分数。组内消维已做,**剩余冗余 = 每请求 g=4 组行**(非 32;H_MAIN=32 是 caller 视角,kernel 契约是 g=4)。

### 修正后改动点
1. ProcessGroupMean:q̄ 累加按组分组 —— 现 q̄[R,g,128] mean(own);改为折叠 acc2[g,128]·w̄[g] 同步累加 → 输出 q̄2[R,128](fp32 累加单次舍入 bf16)。mode0:A/B 探针 caller 侧改算 q̄2 = einsum(wb[r,g], qb[r,g,d])(probe_m1a 同步改)。
2. workspace:qBarWs R*4*128*2B → R*128*2B;wBarWs 删除(tiling.cpp + KH Init + InitProxyGlobalTensor 六处 grep 闭合,含 mode0 绑定分支)。
3. cube:M 维几何从 [R,4] 摊平变 [R](actMBaseSize/mBaseSize/actMBaseSizeAlign 语义链在 CalcRunInfo/SplitCore/workspace 需 grep 闭合);mm1 输出 [R行, s2]。
4. V 主 pass:删 DoScale/DoReduce 调用与 weightsInUb/CopyWeight 链,mm1Res 行直作 sortScore(rNum=1 路径 = DoReduce 的 rNum==1 分支语义)。
5. refine 0 改动;aslk/candidates 契约不变。
6. 验证链照 M3:rebuild(注册面变 → 全量再生)→ probe_m1a 7/7 → precision_gate_full → test.py --probe 四相位 → bench_m3r(R 斜率不变量:塌缩后斜率应仍 ~12-15µs/R 或更优)→ AIME。

### 已知风险
- mBaseSize 语义贯穿 CalcRunInfo(tndBIdxOffset/weightsCoreOffset/indiceOutOffset)/SplitCore/mBaseSizeAlign —— 改一处必须 grep 全消费点(记忆:shared-server-and-field-closure)。
- probe A/B:mode0 现在传 qb[R,4,128]/wb[R,4] → caller 侧预折叠成 qb2[R,128](torch einsum 与预阶段同序同型)。
- M 维塌缩后 M_BASIC_BLOCK 块划分参数可能需要重调(每请求 1 行 → actMBaseSize 大幅变小)。

### ⚠ 契约疑点(动手前必须先解码,2026-10-10 07:50)
- op_host:query = TND [N, H=32, 128](n1Size_=dim1,QUERY_HEAD_NUM_LIMIT);weights 2 维。
- kernel qRowSize = headDim*gSize = 4*128 = 512 元素/行(qHeadNum≡gSize=4)。
- mode0 caller(probe)传 qb=[R,32,128]却被按 [R,4,128] 步进消费且 A/B 位等成立 ——
  说明 32"头"维在 caller/gate 参考里有重解释。**先读 precision_gate 的 torch 参考
  与 pivot_indexer.py 的调用形状,确认 [N,H,D] 里谁被 mean、g=4 与 H=32 的真实关系,
  再定折叠的数学形式**(当前 g=4 → 塌缩仅 4×,非文档假设的 32×;V 侧 DoScale/DoReduce
  收益有限,L 主导的 mm1 不变 → 期望收益需重新评估,可能要转向 s2BaseSize 增大二期)。

### ⛔ 死判决(用户确认,2026-10-10 上午)
**M2 头折叠(压缩后再计算)= 已尝试且已 revert**(见记忆 pivot-kernel-bench-and-gate"fold 已 revert")。失败签名(用户原话要点):
- 回答结果里出现大量重复;
- 输出最前面一直是 65536(索引域上界泄漏为 token);
- 一般只有最后四五个(最长生成的)答案才出现重复/超长。
机理:q̄2 多一次 bf16 舍入 → topk 边界候选微变 → 整网行为漂移,长生成敏感放大。
**单算子门(probe A/B 位等/精度召回门)全绿也守不住 —— 必须整网 AIME 门裁决。**
结论:压缩后再计算的 M2 路线永久关闭,勿再重试。剩余时延杠杆:
s2BaseSize 增大二期(cube 带宽)、refine 91.6µs 优化、窗口/poll 开销摊薄。
