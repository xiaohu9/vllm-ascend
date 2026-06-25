/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather Kernel Entry — Rev 5.3 inplace, aivNum block_dim
 *
 * Inplace contract via OpDef Input("attn_ref")+Output("attn_ref") same-name +
 * _ref suffix → opbuild emits NnopbaseSetRef → framework binds Output GM to
 * Input GM at executor build time.
 *   attn_in / attn_out → SAME GM address (SetRef-guaranteed)
 *   lse_in  / lse_out  → SAME GM address (SetRef-guaranteed)
 * Inner API generated from OpDef passes only 3 input tensors; kernel entry
 * still receives 5 GM_ADDR (3 inputs + 2 outputs) but the two pairs point to
 * identical addresses. mask_num is a 0-d int32 device tensor — kernel reads
 * it at runtime via DataCopyPad (aclgraph-compatible).
 *
 * Launcher (Rev 5.3): default (NO KERNEL_TASK_TYPE_DEFAULT) — host SetBlockDim
 * uses min(aivNum, max(cp_size_, slotCRowsMax)) per user 硬约束 "block_dim 按
 * 实际批次裁剪". SplitCoreCalFor{Rank,Token} divides work; idle AIV-core sees
 * sendRankNum_ == 0 / sendTokenNum_ == 0 → for-loops trivially skip → still
 * participates in launcher-wide SyncAll barriers (no bare return per F17).
 *
 * Reference: mc2/mask_all_to_all_v2/op_kernel/mask_all_to_all_v2.cpp (5-param entry)
 *            mc2/moe_distribute_combine SplitCoreCal paradigm (aivNum launcher)
 */

#include "kernel_operator.h"
#include "allto_all_attn_update_all_gather_tiling.h"
#include "allto_all_attn_update_all_gather_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void allto_all_attn_update_all_gather(
    GM_ADDR attn_in, GM_ADDR lse_in, GM_ADDR mask_num,
    GM_ADDR attn_out, GM_ADDR lse_out,                  // SetRef: == attn_in / lse_in
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(Mc2Tiling::AlltoAllAttnUpdateAllGatherTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

    AlltoAllAttnUpdateAllGather::KernelAlltoAllAttnUpdateAllGather<
        Mc2Tiling::AlltoAllAttnUpdateAllGatherTilingData> op(&pipe);
    op.Init(attn_in, lse_in, mask_num, attn_out, lse_out, &tilingData, contextGM);
    op.Process();
}
