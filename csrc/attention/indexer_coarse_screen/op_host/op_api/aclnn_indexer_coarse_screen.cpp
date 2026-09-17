/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_indexer_coarse_screen.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerIndexerCoarseScreenGetWorkspaceSize(
    const aclTensor *qBar, const aclTensor *wBar, const aclTensor *key,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional, int64_t coarseCount, int64_t hasWindow,
    const aclTensor *candidatesOut, const aclTensor *aslkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerIndexerCoarseScreen(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 const aclrtStream stream);

aclnnStatus aclnnIndexerCoarseScreenGetWorkspaceSize(
        const aclTensor *qBar,
        const aclTensor *wBar,
        const aclTensor *key,
        const aclTensor *actualSeqLengthsQueryOptional,
        const aclTensor *actualSeqLengthsKeyOptional,
        const aclTensor *blockTableOptional,
        int64_t coarseCount,
        int64_t hasWindow,
        const aclTensor *candidatesOut,
        const aclTensor *aslkOut,
        uint64_t *workspaceSize,
        aclOpExecutor **executor)
{
    if (qBar == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "q_bar pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }

    return aclnnInnerIndexerCoarseScreenGetWorkspaceSize(
        qBar, wBar, key, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional,
        blockTableOptional, coarseCount, hasWindow, candidatesOut, aslkOut, workspaceSize, executor);
}

aclnnStatus aclnnIndexerCoarseScreen(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerIndexerCoarseScreen(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
