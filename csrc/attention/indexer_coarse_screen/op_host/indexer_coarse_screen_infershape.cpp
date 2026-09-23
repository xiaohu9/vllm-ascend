/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_coarse_screen_infershape.cpp
 * \brief 输出 shape 推导:固定 query=TND [N,H,D], key=PA_BSND [BlockNum,BlockSize,1,D]
 *        candidates [R, W'], aslk_out [R]
 *        W' = coarse_count + (has_window ? 2g-1 : 0)
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"


using namespace ge;

namespace ops {
constexpr uint32_t QBAR_INDEX = 0;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t CANDIDATES_INDEX = 0;
constexpr uint32_t ASLK_OUT_INDEX = 1;
constexpr uint32_t ATTR_COARSE_COUNT_INDEX = 0;
constexpr uint32_t ATTR_HAS_WINDOW_INDEX = 1;

static ge::graphStatus InferShapeIndexerCoarseScreen(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerCoarseScreen", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape *qBarShape = context->GetInputShape(QBAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qBarShape);
    const gert::Shape *actualSeqQShape = context->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualSeqQShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *coarseCount = attrs->GetInt(ATTR_COARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, coarseCount);
    const int64_t *hasWindow = attrs->GetInt(ATTR_HAS_WINDOW_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hasWindow);
    const int64_t *gSize = attrs->GetInt(2); // ATTR_GROUP_SIZE_INDEX(第 3 个 attr)
    OP_CHECK_NULL_WITH_CONTEXT(context, gSize);
    OP_CHECK_IF((*gSize <= 0) || (*gSize > 16),
               OP_LOGE(context, "group_size must be in (0, 16], but now is %ld.", *gSize),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(qBarShape->GetDimNum() != 3,
               OP_LOGE(context, "Layout TND, q_bar dims (%zu) must be 3!", qBarShape->GetDimNum()),
               return ge::GRAPH_FAILED);
    const int64_t reqNum = actualSeqQShape->GetDim(0);
    OP_CHECK_IF(qBarShape->GetDim(0) != reqNum,
               OP_LOGE(context, "q_bar dim0 (%ld) must equal actual_seq_lengths_query dim0 (%ld).",
                       qBarShape->GetDim(0), reqNum),
               return ge::GRAPH_FAILED);

    gert::Shape *candidatesShape = context->GetOutputShape(CANDIDATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, candidatesShape);
    gert::Shape *aslkOutShape = context->GetOutputShape(ASLK_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, aslkOutShape);

    // 输出 candidates [R, W'](0-based 逻辑 key 位置,-1 终止);实际有效宽度由
    // aslk' 逐行决定,行尾 -1。窗口列 = [aslk-(g-1), aslk+aslq差分) 并集上界 2g-1。
    int64_t outW = *coarseCount + (*hasWindow != 0 ? 2 * *gSize - 1 : 0);
    candidatesShape->SetDimNum(2);
    candidatesShape->SetDim(0, reqNum);
    candidatesShape->SetDim(1, outW);
    // aslk_out [R] 每请求有效候选数(refine 的 actual_seq_lengths_key 直接消费)
    aslkOutShape->SetDimNum(1);
    aslkOutShape->SetDim(0, reqNum);
    OP_LOGI(context->GetNodeName(), "IndexerCoarseScreen InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerCoarseScreen(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerCoarseScreen", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter IndexerCoarseScreen InferDataType impl.");
    // 两个输出恒为 int32
    context->SetOutputDataType(CANDIDATES_INDEX, ge::DT_INT32);
    context->SetOutputDataType(ASLK_OUT_INDEX, ge::DT_INT32);
    OP_LOGI(context->GetNodeName(), "IndexerCoarseScreen InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerCoarseScreen)
    .InferShape(InferShapeIndexerCoarseScreen)
    .InferDataType(InferDataTypeIndexerCoarseScreen);
} // namespace ops
