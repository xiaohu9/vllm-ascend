/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather Tiling Data Structure (Kernel-side) — Rev 5.3 inplace
 *
 * Inplace: attn_ref == attn (same GM address); lse_ref == lse.
 * Active rows [0, b0_total) Phase A→B→C; Inactive rows pass-through (kernel 不动).
 *
 * CRITICAL: Mc2InitTiling and Mc2CcTiling MUST be at the front of this struct.
 * Layout verified against mc2/mask_all_to_all_v2 production code (Rev 1.0.2).
 *
 * Field provenance:
 *   v2 共有: totalT / hDim / lseDim / groupSize / attnLineBytes / lseLineBytes
 *           / attnRowSize / lseRowSize / rowSize
 *   Rev 5.3 新增 (本算子专属):
 *     - aivNum                 — kernel SplitCoreCalForRank/Token 用
 *     - slotABytesPerRank      — Phase A peermem slot 每 rank 字节数
 *     - slotCBytesPerRank      — Phase C peermem slot 每 rank 字节数
 *     - slotAOffsetInWin       — = 0
 *     - slotCOffsetInWin       — = groupSize · slotABytesPerRank
 *     - slotCRowsMax           — = totalT / groupSize  (slot C 上界)
 *     - maxRowsPerSubtile      — Phase A/C UB ping-pong 单半切行上界
 *     - numTiles / maxTileB0   — Phase A/C 切 tile 元信息 (与 v2 同义)
 *
 * b0 / b1 维度同 v2: 仍由 kernel 在 Init 期间 DataCopyPad 读 mask_num 派生，
 *                  host tiling 不读 device 0-d tensor (aclgraph 静态图限制).
 */

#pragma once

#include "kernel_tiling/kernel_tiling.h"

namespace Mc2Tiling {

// socParam 不再存在(vllm-ascend csrc 移除 mc2_tiling_utils.h 依赖),
// kernel 层从未读 commAlg/isA3 — 死代码已删.

class AlltoAllAttnUpdateAllGatherTilingData {
public:
    // ===== HCCL Tiling (MUST be at front!) =====
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling   mc2CcTiling;

    // ===== Shape =====
    uint32_t totalT;           // T·cp        (attn.shape[0])
    uint32_t hDim;             // n/cp · D    (attn.shape[1])
    uint32_t lseDim;           // n/cp        (lse.shape[1])
    uint32_t groupSize;        // cp_size

    // ===== Block Dim (Rev 5.3) =====
    uint32_t aivNum;           // = ascendcPlatform.GetCoreNumAiv()  (SplitCoreCal 用)

    // ===== Per-row layout (32B aligned) =====
    uint32_t attnLineBytes;    // hDim   * sizeof(bf16)
    uint32_t lseLineBytes;     // lseDim * sizeof(fp32)
    uint32_t attnRowSize;      // AlignUp32(attnLineBytes)
    uint32_t lseRowSize;       // AlignUp32(lseLineBytes)
    uint32_t rowSize;          // attnRowSize + lseRowSize

    // ===== Peermem slot layout (Rev 5.3) =====
    uint32_t slotCRowsMax;             // = totalT / groupSize
    uint64_t slotABytesPerRank;        // = totalT       · rowSize
    uint64_t slotCBytesPerRank;        // = slotCRowsMax · rowSize
    uint64_t slotAOffsetInWin;         // = 0
    uint64_t slotCOffsetInWin;         // = groupSize · slotABytesPerRank

    // ===== Tile control (Phase A/C UB ping-pong, Rev 5.3) =====
    uint32_t maxRowsPerSubtile;        // single ping-pong half row 上界
    uint32_t numTiles;                 // 由 kernel runtime 根据 b0_total 派生
    uint32_t maxTileB0;                // 单 tile B0 上界
};

}  // namespace Mc2Tiling
