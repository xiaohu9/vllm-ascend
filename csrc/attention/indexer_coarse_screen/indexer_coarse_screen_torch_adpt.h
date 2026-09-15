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
#ifndef INDEXER_COARSE_SCREEN_TORCH_ADPT_H
#define INDEXER_COARSE_SCREEN_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> construct_indexer_coarse_screen_output_tensor(
    const at::Tensor& query, const at::Tensor& row_weights, int64_t coarse_count, int64_t has_window)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < row_weights.sizes().size(); i++) {
        TORCH_CHECK(row_weights.size(i) > 0,
                    "All values within row_weights' shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", row_weights.size(i));
    }
    TORCH_CHECK(coarse_count > 0,
                "coarse count should be greater than 0, but now is ",
                coarse_count);
    // 固定 TND query + PA_BSND key:候选 [R, W'],W' = coarse + (has_window ? 2g-1 : 0)
    const int64_t req_num = row_weights.size(DIM_0);
    const int64_t group_size = row_weights.size(DIM_1);
    int64_t out_w = coarse_count;
    if (has_window != 0) {
        out_w = coarse_count + 2 * group_size - 1;
    }
    output_size = {req_num, out_w};

    at::Tensor candidates_out = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor aslk_out = at::empty({req_num}, query.options().dtype(at::kInt));
    return std::tuple<at::Tensor, at::Tensor>(candidates_out, aslk_out);
}

std::tuple<at::Tensor, at::Tensor> npu_indexer_coarse_screen(
    const at::Tensor& query, const at::Tensor& weights, const at::Tensor& row_weights,
    const at::Tensor& key,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const at::Tensor& block_table, int64_t coarse_count, int64_t has_window)
{
    // 图安全:仅做 shape/dtype 存在性检查,捕获期不读张量数据
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
    TORCH_CHECK(row_weights.numel() > 0, "Tensor row_weights is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(block_table.numel() > 0, "Tensor block_table is empty.");

    auto coarse_screen_output = construct_indexer_coarse_screen_output_tensor(
        query, row_weights, coarse_count, has_window);
    at::Tensor candidates_out = std::get<0>(coarse_screen_output);
    at::Tensor aslk_out = std::get<1>(coarse_screen_output);

    EXEC_NPU_CMD(aclnnIndexerCoarseScreen, query, weights, row_weights, key,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 coarse_count, has_window, candidates_out, aslk_out);

    return std::tuple<at::Tensor, at::Tensor>(candidates_out, aslk_out);
}
}  // namespace vllm_ascend

#endif  // INDEXER_COARSE_SCREEN_TORCH_ADPT_H
