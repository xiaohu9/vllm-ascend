/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ALLTO_ALL_ATTN_UPDATE_ALL_GATHER_TORCH_ADPT_H
#define ALLTO_ALL_ATTN_UPDATE_ALL_GATHER_TORCH_ADPT_H
namespace vllm_ascend {
 
void npu_allto_all_attn_update_all_gather(
    at::Tensor &attn,
    at::Tensor &lse,
    const at::Tensor &mask_num,
    c10::string_view group,
    int64_t group_size)
{
    // Inplace operator: attn and lse are both input and output
    std::string group_str(group);
    char *group_ptr = group_str.data();

    EXEC_NPU_CMD(aclnnAlltoAllAttnUpdateAllGather,
        attn, lse, mask_num,
        group_ptr, group_size);
}
 
}
#endif