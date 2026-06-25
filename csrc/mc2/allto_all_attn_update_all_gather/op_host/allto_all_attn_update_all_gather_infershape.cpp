/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather InferShape & InferDataType — Rev 5.3 inplace
 *
 * Inplace: Input and Output share the same name (attn_ref, lse_ref).
 * Output shapes = input shapes — kernel 只做 row 排列 + LSE-weighted reduce + head
 * AllGather, 张量形状不变.
 *
 * Pattern: 照抄 mc2/mask_all_to_all_v2/op_host/mask_all_to_all_v2_infershape.cpp:1-47.
 * CANN 8.5.1: IMPL_OP_INFERSHAPE(Op).InferShape(fn).InferDataType(fn)
 *   - namespace ops; using namespace ge for graphStatus enums.
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeAlltoAllAttnUpdateAllGather(gert::InferShapeContext* context) {
    auto attnShape = context->GetOutputShape(0);    // attn_ref [totalT, n_per_cp · D]
    auto lseShape  = context->GetOutputShape(1);    // lse_ref  [totalT, n_per_cp]

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        attnShape == nullptr || lseShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Output shapes match input shapes exactly (in-place, no shape change).
    *attnShape = *context->GetInputShape(0);
    *lseShape  = *context->GetInputShape(1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllAttnUpdateAllGather(gert::InferDataTypeContext* context) {
    // attn_ref: BF16 (same as Input attn_ref)
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // lse_ref:  FLOAT (same as Input lse_ref)
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllAttnUpdateAllGather)
    .InferShape(InferShapeAlltoAllAttnUpdateAllGather)
    .InferDataType(InferDataTypeAlltoAllAttnUpdateAllGather);

}  // namespace ops
