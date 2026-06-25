/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather ACLNN Two-Stage Interface Declaration
 *
 * Pattern: MC2 fused operator ACLNN, mirrors aclnn_mask_all_to_all_v2.h
 *          + inplace pattern (Input("X_ref") / Output("X_ref") same-name +
 *          _ref suffix → opbuild emits NnopbaseSetRef → inner API drops
 *          duplicate Output params).
 *
 * Operator: in-place fused {alltoall + cross-cp LSE-weighted attn update +
 *           head-AllGather + permute} for CP (context parallel).
 *
 *   attn               : [T·cp, n/cp · D]  bf16   (Input & Output, same tensor)
 *   lse                : [T·cp, n/cp]      fp32   (Input & Output, same tensor)
 *   mask_num           : []  int32  (0-d; per-rank active token count)
 *   group, group_size  : HCCL group descriptor (group_size = cp ∈ {1,2,4,8,16})
 *
 * Active rows [0, mask_num × cp): undergo full three-phase fusion in place.
 * Inactive rows [mask_num × cp, T·cp): kernel does not touch them
 *   (already in place by caller contract; kernel respects pass-through).
 */

#pragma once

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stage 1: Compute workspace size and validate parameters.
 *
 * @param attn       [T·cp, n/cp · D] bf16   inplace 输入兼输出
 * @param lse        [T·cp, n/cp]     fp32   inplace 输入兼输出
 * @param mask_num   [] int32, value = mask_per_rank (active token count, per-rank)
 * @param group      HCCL communication group name
 * @param group_size cp size (1, 2, 4, 8, 16)
 * @param workspaceSize [out]
 * @param executor      [out]
 */
ACLNN_API aclnnStatus aclnnAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
    const aclTensor *attn,
    const aclTensor *lse,
    const aclTensor *mask_num,
    char *group,
    int64_t group_size,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * Stage 2: Execute.
 */
ACLNN_API aclnnStatus aclnnAlltoAllAttnUpdateAllGather(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
