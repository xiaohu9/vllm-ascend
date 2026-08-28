/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_refine_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"


using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;

static ge::graphStatus InferShapeIndexerRefine(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerRefine", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    const int64_t *seleced_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seleced_count);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    // refine 固定 query=TND, key=PA_BSND
    OP_CHECK_IF(
        inputLayoutQueryPtrStr != "TND",
        OP_LOGE(context, "The attr layout_query should be TND, but got %s.", inputLayoutQueryPtrStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputLayoutKeyPtrStr != "PA_BSND",
        OP_LOGE(context, "The attr layout_key should be PA_BSND, but got %s.", inputLayoutKeyPtrStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        queryShape->GetDimNum() != 3,
        OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    // 输出 TND 布局 [T, N2(恒 1), refineCount]
    sparseIndicesShape->SetDimNum(3);
    sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim T
    sparseIndicesShape->SetDim(1, keyShape->GetDim(2));   // 1:Dim N(PA_BSND key [BlockNum,BlockSize,N,D])
    sparseIndicesShape->SetDim(2, *seleced_count);        // 2:Dim K(refineCount)
    OP_LOGI(context->GetNodeName(), "IndexerRefine InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerRefine(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerRefine", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter IndexerRefine InferDataType impl.");
    // 输出 topk_indices 恒为 int32
    context->SetOutputDataType(0, ge::DT_INT32);
    OP_LOGI(context->GetNodeName(), "IndexerRefine InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerRefine)
    .InferShape(InferShapeIndexerRefine)
    .InferDataType(InferDataTypeIndexerRefine);
} // namespace ops
