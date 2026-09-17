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
 * \file indexer_coarse_screen_def.cpp
 * \brief PIVOT 粗筛算子原型:组加权均值代理(M1)+ 全前缀 top-coarse_count(M2)
 *        + 局部窗口注入(M2.5,has_window 门控,双输出 candidates'/aslk')
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class IndexerCoarseScreen : public OpDef {
public:
    explicit IndexerCoarseScreen(const char *name) : OpDef(name)
    {
        // 2026-09-17 M1 出核:组均值由 caller 侧算好传入(原 query/weights/row_weights
        // 三输入的内核内 M1 是仓内孤立的 AIV写→AIC读 模式,910B 上 SyncAll 不可靠)
        this->Input("q_bar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("w_bar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // aslq/aslk/block_table 声明 OPTIONAL 仅为对齐 indexer_refine 原型口径;
        // tiling GetAndCheckOptionalInput 实际强制必填(TND+PA_BSND 契约),torch schema 侧必填
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("candidates")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("aslk_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Attr("coarse_count").AttrType(OPTIONAL).Int(4096); // 4096:候选预算(论文粗筛宽度)
        this->Attr("has_window").AttrType(OPTIONAL).Int(1);      // 1:输出阶段并集注入局部窗口(生产)
        this->Attr("group_size").AttrType(OPTIONAL).Int(4);      // g:组内 query 数(窗口宽 2g-1;
                                                                 // decode=MTP g,prefill=PREFILL_GROUP)
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        // 仅 A3(910B)目标:kernel 只有 arch22,不加 950(A5 arch35)——
        // 声明而无对应 arch 目录会触发 opbuild "Invalid socVersion" 校验失败(同 indexer_refine)
    }
};
OP_ADD(IndexerCoarseScreen);
} // namespace ops
