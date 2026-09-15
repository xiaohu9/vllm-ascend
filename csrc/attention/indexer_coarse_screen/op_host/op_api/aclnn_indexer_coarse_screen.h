/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_INDEXER_COARSE_SCREEN_H
#define ACLNN_INDEXER_COARSE_SCREEN_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnIndexerCoarseScreenGetWorkspaceSize calculates the workspace size based on the
 *        specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default")))
aclnnStatus aclnnIndexerCoarseScreenGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *weights,
    const aclTensor *rowWeights,
    const aclTensor *key,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    int64_t coarseCount,
    int64_t hasWindow,
    const aclTensor *candidatesOut,
    const aclTensor *aslkOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnIndexerCoarseScreen is used to perform calculations.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnIndexerCoarseScreen(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_INDEXER_COARSE_SCREEN_H
