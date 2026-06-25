/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather Host Tiling — Rev 5.3 inplace, block_dim 按实际批次裁剪
 *
 * Rev 5.3 关键变更（vs v2 mask_all_to_all_v2_tiling.cpp）：
 *   (1) block_dim 按实际批次裁剪（不无脑取 aivNum）：
 *         Phase A/C 切核上界 = cp_size_      （仿 moe_distribute_combine SplitCoreCalForRank）
 *         Phase B   切核上界 = slotCRowsMax   （= totalT / cp_size_，token 切核上界）
 *         block_dim = min(aivNum, max(cp_size_, slotCRowsMax))
 *       —— 多分核没活干只是给 SyncAll 凑数,反而拖慢屏障；按批次裁是用户硬约束
 *   (2) 新增 slot 布局字段：slotABytesPerRank / slotCBytesPerRank
 *                          / slotAOffsetInWin / slotCOffsetInWin / slotCRowsMax
 *   (3) 新增 tile 控制字段：maxRowsPerSubtile / numTiles / maxTileB0
 *       numTiles / maxTileB0 由 kernel runtime 派生 (mask_num device-only)
 *   (4) Phase B (LSE-weighted reduce) per-token UB 容量校验
 *
 * 与 v2 相同部分（90%）：
 *   - 2D shape 校验、attr 校验、totalT % cp_size 校验
 *   - rowSize = AlignUp32(attnLineBytes) + AlignUp32(lseLineBytes)
 *   - peermem window 容量校验（slot A + slot C 联合占用）
 *   - SoC topology 解析 (FULL_MESH / DOUBLE_RING)
 *   - Mc2CcTilingConfig: commtype = 8 (ALLTOALLV), algConfig = "AlltoAll=level0:fullmesh;level1:pairwise"
 *     (vllm-ascend csrc 邻居 dispatch_ffn_combine 同款, 无 SetCommEngine 调用)
 *   - workspaces[0] = GetLibApiWorkSpaceSize()
 *   - SetScheduleMode(1)
 *
 * 详见 plans/mask_all_to_all_v2_attn_update_dev_plan.md §3.2 (Rev 5.3).
 */

#include "vector"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
// vllm-ascend csrc 适配（对齐邻居 dispatch_ffn_combine_tiling.cpp）：
//   - 移除 ops-transformer 内部头 mc2_hcom_topo_info.h / mc2_log.h / ops_utils.h /
//     tiling/mc2_tiling_utils.h（csrc 树下均不存在）
//   - mc2_log.h          → tiling_base/error_log.h (OP_TILING_CHECK/OP_LOGE/OP_LOGD)
//   - platform_infos_def → tiling/platform/platform_ascendc.h
//   - mc2_tiling_utils.h 符号替代:
//       * Mc2TilingUtils::GetMaxWindowSize() → 本地 GetMaxWindowSize() 读 HCCL_BUFFSIZE
//         (与 dispatch_ffn_combine 同款,单位 MB,默认 200MB)
//       * Mc2TilingUtils::GetCommSets / COMM_MESH / COMM_ALG_* / socParam → 删除
//         (kernel 端从未读 socParam 字段,死代码;algConfig 由 fullmesh 字符串 hardcode,
//          A3 拓扑切换由 HCCL 内部基于 group 自动识别)
//       * AicpuComType::HCCL_CMD_ALLTOALLV → 常量 8 (邻居 OP_TYPE_ALL_TO_ALL 同款)
//       * mc2tiling::AIV_ENGINE / SetCommEngine → 删除(邻居均无此调用)
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/allto_all_attn_update_all_gather_tiling.h"

// ops-transformer 内部宏在 vllm-ascend 树下无定义，退化为 OP_LOGE（行为等价：
// 仅缺 REPORT_INNER_ERR_MSG 的 E89999 错误码上报，不影响功能）。
#ifndef VECTOR_INNER_ERR_REPORT_TILING
#define VECTOR_INNER_ERR_REPORT_TILING(op_name, err_msg, ...) \
    OP_LOGE(op_name, err_msg, ##__VA_ARGS__)
#endif

namespace {
// 与 dispatch_ffn_combine_tiling.cpp::GetMaxWindowSize 行为对齐.
// HCCL_BUFFSIZE 单位 MB,默认 200MB.
constexpr const char* HCCL_BUFFSIZE_ENV = "HCCL_BUFFSIZE";
constexpr uint64_t MB_TO_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t DEFAULT_HCCL_BUFFSIZE_MB = 200ULL;

static uint64_t GetMaxWindowSize() {
    uint64_t windowSizeMb = DEFAULT_HCCL_BUFFSIZE_MB;
    const char* env = std::getenv(HCCL_BUFFSIZE_ENV);
    if (env != nullptr) {
        try {
            uint64_t v = std::stoull(env);
            if (v > 0 && v <= std::numeric_limits<uint16_t>::max()) {
                windowSizeMb = v;
            }
        } catch (const std::exception&) {
            // ignore — fallback to default
        }
    }
    return windowSizeMb * MB_TO_BYTES;
}
}  // namespace

using namespace AscendC;
using namespace ge;

namespace optiling {

// 32B-align (peermem slot stride requirement)
static inline int64_t AlignUp32(int64_t bytes) {
    return (bytes + 31) / 32 * 32;
}

// Phase A/C ping-pong 单半 = 80KB; 单 subtile 装 MAX_ROWS_PER_SUBTILE=2 行
// → rowSize ≤ 80KB / 2 = 40KB. (kernel.h USED_UB_SIZE = 160 KB)
static constexpr int64_t MAX_ROW_SIZE_BYTES         = 40 * 1024;

// Phase B per-token reduce UB 总容量上界 = USED_UB_SIZE = 160 KB.
// Layout (一次性方案,见 kernel.h PhaseBReduce):
//   lsePad=AlignUp8(lseDim), spPadAlign=AlignUp64(cp·lsePad), dHead=hDim/lseDim
//   LSE 全 lane 一次 reduce；attn 按 PHASE_B_LANE_BLOCK lane 流式 merge/write。
//   ubInFp32 cp·laneBlock·dHead·4 | ubLseFp32/ubLseExp/ubNegInf spPadAlign·4 each
//   | ubLseM/Sum/Out lsePad·4·3 | ubAccFp32 laneBlock·dHead·4
//   | ubMaskU8 spPadAlign | ubAttnBf cp·laneBlock·dHead·2 | ubOutBf laneBlock·dHead·2
static constexpr int64_t MAX_REDUCE_UB_BYTES        = 160 * 1024;
static constexpr int64_t PHASE_B_LANE_BLOCK         = 8;

// 单 subtile 行数（与 v2 USED_UB_HALF / (2·rowSize) 一致；hardcode 2）
static constexpr uint32_t MAX_ROWS_PER_SUBTILE      = 2;

static ge::graphStatus AlltoAllAttnUpdateAllGatherTilingFunc(gert::TilingContext *context) {
    auto tilingData = context->GetTilingData<Mc2Tiling::AlltoAllAttnUpdateAllGatherTilingData>();

    // ---------- 1. Tensor shapes ----------
    const gert::StorageShape *attnShape = context->GetInputShape(0);   // [totalT, n/cp·D]
    const gert::StorageShape *lseShape  = context->GetInputShape(1);   // [totalT, n/cp]
    OP_TILING_CHECK(attnShape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "attn_ref shape is nullptr"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(lseShape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "lse_ref shape is nullptr"),
        return ge::GRAPH_FAILED);

    const auto &attnStorage = attnShape->GetStorageShape();
    const auto &lseStorage  = lseShape->GetStorageShape();
    OP_TILING_CHECK(attnStorage.GetDimNum() != 2,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "attn_ref must be 2D, got %zu dims", attnStorage.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(lseStorage.GetDimNum() != 2,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "lse_ref must be 2D, got %zu dims", lseStorage.GetDimNum()),
        return ge::GRAPH_FAILED);

    uint32_t totalT = static_cast<uint32_t>(attnStorage.GetDim(0));
    uint32_t hDim   = static_cast<uint32_t>(attnStorage.GetDim(1));
    uint32_t lseDim = static_cast<uint32_t>(lseStorage.GetDim(1));

    OP_TILING_CHECK(static_cast<uint32_t>(lseStorage.GetDim(0)) != totalT,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "lse_ref.dim(0)=%u must equal attn_ref.dim(0)=%u",
            static_cast<uint32_t>(lseStorage.GetDim(0)), totalT),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(lseDim == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "lse_ref.dim(1) must be > 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hDim % lseDim != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "attn_ref.dim(1)=%u must be a multiple of lse_ref.dim(1)=%u (D)",
            hDim, lseDim),
        return ge::GRAPH_FAILED);

    // ---------- 2. Attrs ----------
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "attrs is nullptr"),
        return ge::GRAPH_FAILED);
    auto groupPtr     = attrs->GetAttrPointer<char>(0);
    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(1);
    OP_TILING_CHECK(groupPtr == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group attr is nullptr"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(groupSizePtr == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group_size attr is nullptr"),
        return ge::GRAPH_FAILED);

    uint32_t groupSize = static_cast<uint32_t>(*groupSizePtr);
    OP_TILING_CHECK(groupSize == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group_size must be > 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(totalT % groupSize != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "totalT=%u must be a multiple of group_size=%u (CP)", totalT, groupSize),
        return ge::GRAPH_FAILED);

    // ---------- 3. Tiling struct (shape) ----------
    tilingData->totalT        = totalT;
    tilingData->hDim          = hDim;
    tilingData->lseDim        = lseDim;
    tilingData->groupSize     = groupSize;
    tilingData->attnLineBytes = hDim   * static_cast<uint32_t>(sizeof(uint16_t));   // bf16
    tilingData->lseLineBytes  = lseDim * static_cast<uint32_t>(sizeof(float));      // fp32
    tilingData->attnRowSize   = static_cast<uint32_t>(AlignUp32(tilingData->attnLineBytes));
    tilingData->lseRowSize    = static_cast<uint32_t>(AlignUp32(tilingData->lseLineBytes));
    tilingData->rowSize       = tilingData->attnRowSize + tilingData->lseRowSize;

    OP_TILING_CHECK(static_cast<int64_t>(tilingData->rowSize) > MAX_ROW_SIZE_BYTES,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "rowSize=%u exceeds Phase A/C UB-per-subtile cap %ld "
            "(attnRow=%u + lseRow=%u). Reduce hDim or lseDim.",
            tilingData->rowSize, MAX_ROW_SIZE_BYTES,
            tilingData->attnRowSize, tilingData->lseRowSize),
        return ge::GRAPH_FAILED);

    // Phase B per-token reduce UB 容量校验 (LSE 全 lane + attn lane-streaming,见 kernel.h PhaseBReduce)
    int64_t cp_int     = static_cast<int64_t>(groupSize);
    int64_t hDim_int   = static_cast<int64_t>(hDim);
    int64_t lseDim_int = static_cast<int64_t>(lseDim);
    int64_t lsePad_int = ((lseDim_int + 7) / 8) * 8;
    int64_t dHead_int  = hDim_int / lseDim_int;
    int64_t laneBlock_int = PHASE_B_LANE_BLOCK;
    int64_t blockElems_int = laneBlock_int * dHead_int;
    int64_t spPad_int      = cp_int * lsePad_int;
    int64_t spPadAlign_int = ((spPad_int + 63) / 64) * 64;
    int64_t reduceUbBytes =
          cp_int * blockElems_int * 4                 // ubInFp32
        + spPadAlign_int * 4                          // ubLseFp32
        + spPadAlign_int * 4                          // ubLseExp
        + lsePad_int * 4 * 3                          // ubLseM + ubLseSum + ubLseOut
        + blockElems_int * 4                          // ubAccFp32
        + spPadAlign_int * 4                          // ubNegInf
        + spPadAlign_int                              // ubMaskU8
        + cp_int * blockElems_int * 2                 // ubAttnBf
        + blockElems_int * 2;                         // ubOutBf
    OP_TILING_CHECK(reduceUbBytes > MAX_REDUCE_UB_BYTES,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "Phase B reduce UB bytes %ld > USED_UB_SIZE %ld "
            "(cp=%ld, hDim=%ld, lseDim=%ld, lsePad=%ld, spPadAlign=%ld, dHead=%ld, laneBlock=%ld). "
            "Reduce hDim, lseDim, laneBlock, or cp_size.",
            reduceUbBytes, MAX_REDUCE_UB_BYTES, cp_int, hDim_int, lseDim_int,
            lsePad_int, spPadAlign_int, dHead_int, laneBlock_int),
        return ge::GRAPH_FAILED);

    // ---------- 4. Peermem slot 布局 ----------
    // slot A: cp 个 rank 区段，每段 totalT 行（按 mask_num 上界 b0_total ≤ totalT 预留）
    // slot C: cp 个 rank 区段，每段 totalT/cp 行（head-AllGather 后每 rank 仅持自己的子表）
    tilingData->slotCRowsMax        = totalT / groupSize;
    tilingData->slotABytesPerRank   = (uint64_t)totalT * tilingData->rowSize;
    tilingData->slotCBytesPerRank   = (uint64_t)tilingData->slotCRowsMax * tilingData->rowSize;
    tilingData->slotAOffsetInWin    = 0ULL;
    tilingData->slotCOffsetInWin    = (uint64_t)groupSize * tilingData->slotABytesPerRank;

    int64_t neededWinBytes = static_cast<int64_t>(
        groupSize * tilingData->slotABytesPerRank +
        groupSize * tilingData->slotCBytesPerRank);
    uint64_t maxWinBytes = GetMaxWindowSize();
    OP_TILING_CHECK(static_cast<uint64_t>(neededWinBytes) > maxWinBytes,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "peermem window overflow: needed %ld > max %lu "
            "(cp=%u totalT=%u rowSize=%u slotA/rk=%lu slotC/rk=%lu). "
            "Increase HCCL_BUFFSIZE or reduce D/totalT.",
            neededWinBytes, maxWinBytes, groupSize, totalT,
            tilingData->rowSize,
            (unsigned long)tilingData->slotABytesPerRank,
            (unsigned long)tilingData->slotCBytesPerRank),
        return ge::GRAPH_FAILED);

    // ---------- 5. HCCL window allocation (no collective is ever issued) ----------
    // commtype = ALLTOALLV(8); A3 拓扑(910_93) HCCL 内部按 group 自动识别,
    // algConfig 字符串与邻居 dispatch_ffn_combine 同款.
    // 注:删除原 SetCommEngine(AIV_ENGINE) 调用 — vllm-ascend csrc 邻居均无此调用,
    //    Mc2CcTilingConfig 默认引擎已满足.
    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    uint32_t commtype = 8U;  // HCCL_CMD_ALLTOALLV
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupPtr, commtype, algConfig);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "GetTiling mc2InitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "GetTiling mc2CcTiling failed"),
        return ge::GRAPH_FAILED);

    // ---------- 6. Workspace ----------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get workspace failed"),
        return ge::GRAPH_FAILED);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    // ---------- 7. Block dim 按实际批次裁剪 (Rev 5.3 用户硬约束) ----------
    //   Phase A/C 切核上界 = cp_size_     (SplitCoreCalForRank: cp 个 dst rank)
    //   Phase B   切核上界 = slotCRowsMax (SplitCoreCalForToken: 上界 mask_per_rank ≤ totalT/cp)
    //   block_dim = min(aivNum, max(cp_size_, slotCRowsMax))
    // 多余核只在 SyncAll 凑数 → 拖慢屏障; 按批次裁严格符合"实际工作量上界".
    uint32_t aivNum   = ascendcPlatform.GetCoreNumAiv();
    uint32_t needRank = groupSize;
    uint32_t needTok  = tilingData->slotCRowsMax;
    uint32_t needMax  = (needRank > needTok) ? needRank : needTok;
    uint32_t usedAiv  = (needMax < aivNum) ? needMax : aivNum;

    OP_TILING_CHECK(aivNum < groupSize,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "aivNum=%u must be >= groupSize=%u", aivNum, groupSize),
        return ge::GRAPH_FAILED);

    tilingData->aivNum            = usedAiv;
    tilingData->maxRowsPerSubtile = MAX_ROWS_PER_SUBTILE;
    // numTiles / maxTileB0 由 kernel Init 在读取 mask_num 后派生 — host 无法读 device 0-d tensor.
    tilingData->numTiles  = 0;
    tilingData->maxTileB0 = 0;

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(usedAiv, 0, usedAiv);
    context->SetBlockDim(blockDim);

    // SetScheduleMode(1) batch mode — 必须，因 SyncAll 需 launcher 等齐全部核.
    context->SetScheduleMode(1);

    OP_LOGD(context->GetNodeName(),
        "AlltoAllAttnUpdateAllGather tiling: totalT=%u hDim=%u lseDim=%u groupSize=%u "
        "attnRow=%u lseRow=%u rowSize=%u slotA/rk=%lu slotC/rk=%lu "
        "physAivNum=%u usedAiv=%u (needRank=%u needTok=%u) blockDim=%u",
        totalT, hDim, lseDim, groupSize,
        tilingData->attnRowSize, tilingData->lseRowSize, tilingData->rowSize,
        (unsigned long)tilingData->slotABytesPerRank,
        (unsigned long)tilingData->slotCBytesPerRank,
        aivNum, usedAiv, needRank, needTok, blockDim);
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllAttnUpdateAllGatherCompileInfo {};

static ge::graphStatus TilingParseForAlltoAllAttnUpdateAllGather(
    [[maybe_unused]] gert::TilingParseContext *context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AlltoAllAttnUpdateAllGather)
    .Tiling(AlltoAllAttnUpdateAllGatherTilingFunc)
    .TilingParse<AlltoAllAttnUpdateAllGatherCompileInfo>(TilingParseForAlltoAllAttnUpdateAllGather);

}  // namespace optiling
