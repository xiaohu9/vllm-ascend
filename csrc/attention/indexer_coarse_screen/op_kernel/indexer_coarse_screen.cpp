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
 * \file indexer_coarse_screen.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "indexer_coarse_screen_template_tiling_key.h"

// 仅 arch22(910b/910_93):def 未注册 950,无 arch35 内核(同 indexer_refine 的目标约束)
#include "arch22/indexer_coarse_screen_kernel.h"

using namespace LIKernel;

#define INVOKE_CS_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<IndexerCoarseScreenType<__VA_ARGS__>> op;                                                        \
        GET_TILING_DATA_WITH_STRUCT(IndexerCoarseScreenTilingData, tiling_data_in, tiling);                            \
        const IndexerCoarseScreenTilingData *__restrict tiling_data = &tiling_data_in;                                 \
        op.Init(qBar, wBar, key, actualSeqLengthsQ, actualSeqLengths, blocktable, candidatesOut,              \n                aslkOut, user, tiling_data, &tPipe);                                                                   \
        op.Process();                                                                                                  \
    } while (0)

// 入口形参序必须与 op def 输入序一致(query, weights, row_weights, key, aslq, aslk, block_table,
// candidates, aslk_out, workspace, tiling)——aclnnInner 按 def 顺序传 GM 指针。
// NPU 实测教训(2026-09-15):曾按 refine 惯例写成 (query, key, weights, rowWeights, ...),
// 导致 key/weights/row_weights 三槽错位 —— M1 的 rw 读成 key 字节、主 pass 的 key 读成
// weights,分数与 query 无关且随分配非确定(A6 位级 dump 定位)。
template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void indexer_coarse_screen(__gm__ uint8_t *qBar, __gm__ uint8_t *wBar,
                                                 __gm__ uint8_t *key,
                                                 __gm__ uint8_t *actualSeqLengthsQ,
                                                 __gm__ uint8_t *actualSeqLengths,
                                                 __gm__ uint8_t *blocktable, __gm__ uint8_t *candidatesOut,
                                                 __gm__ uint8_t *aslkOut,
                                                 __gm__ uint8_t *workspace,
                                                 __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (DT_Q == LI_TPL_FP16 && DT_K == LI_TPL_FP16 && DT_OUT == LI_TPL_INT32) {
        INVOKE_CS_NO_KFC_OP_IMPL(IndexerCoarseScreenKernel, half, half, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG);
    } else {
        INVOKE_CS_NO_KFC_OP_IMPL(IndexerCoarseScreenKernel, bfloat16_t, bfloat16_t, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG);
    }
}
