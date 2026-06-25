/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather OpDef Registration (Rev 5.3 inplace, _ref suffix variant)
 *
 * Operator: fused {AlltoAll + LSE-weighted attention update + head AllGather}
 *           in-place on attn / lse along CP (context parallel) communication group.
 *
 *   attn_ref [totalT, n_per_cp · D] bf16   (Input & Output, same GM address — inplace)
 *   lse_ref  [totalT, n_per_cp]     fp32   (Input & Output, same GM address — inplace)
 *   mask_num []                     int32  (mask_per_rank; b0_total = mask_per_rank · cp_size)
 *
 * Active rows [0, b0_total):
 *   Phase A peermem-pack to peer rank slot
 *   Phase B local LSE-weighted reduce per token
 *   Phase C peermem head-AllGather (反向 stride) back to attn/lse
 * Inactive rows [b0_total, totalT):
 *   走 inplace pass-through — kernel 不读不写, opbuild SetRef 保证 input/output
 *   共享 GM 地址 → caller 现状即输出.
 *
 * Inplace 范式参考: mc2/mask_all_to_all_v2/op_host/mask_all_to_all_v2_def.cpp:32-91
 *   Input("X_ref") + Output("X_ref") 同名 + _ref 后缀 = inplace 契约 (CANN 8.5.1
 *   opbuild 仅在带 _ref 后缀的同名 Input/Output 时发射 NnopbaseSetRef).
 *
 * 命名风格参考: mc2/allto_all_all_gather_batch_mat_mul/op_host/allto_all_all_gather_batch_mat_mul_def.cpp
 *   全 snake_case 文件/目录, CamelCase class 用 `Allto` (非 `AllTo`), 无数字
 *   → 避免 opbuild snake 转换在数字-字母边界插入下划线导致 autogen 文件名失配.
 *
 * CANN 8.5.1 范式: explicit OpDef(const char *name), OpAICoreConfig chain,
 *                  OP_ADD outside class, MC2().HcclGroup("group").
 */

#include "register/op_def_registry.h"

namespace ops {

class AlltoAllAttnUpdateAllGather : public OpDef {
public:
    explicit AlltoAllAttnUpdateAllGather(const char *name) : OpDef(name) {
        // ===== Inputs =====
        // attn_ref / lse_ref: 同名 Input + Output + _ref 后缀 = inplace SetRef gate
        this->Input("attn_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lse_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Outputs (inplace: same name as Inputs, with _ref suffix) =====
        this->Output("attn_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lse_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Attributes =====
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("group_size").AttrType(REQUIRED).Int();

        // ===== AICore Configuration =====
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        // ===== MC2 Communication Domain Declaration =====
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(AlltoAllAttnUpdateAllGather);

}  // namespace ops
