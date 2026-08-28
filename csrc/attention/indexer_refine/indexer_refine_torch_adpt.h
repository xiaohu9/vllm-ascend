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
#ifndef INDEXER_REFINE_TORCH_ADPT_H
#define INDEXER_REFINE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor construct_indexer_refine_output_tensor(
    const at::Tensor& query, const at::Tensor& key, int64_t sparse_count,
    const std::string& query_layout_str, const std::string& key_layout_str)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0,
                    "All values within key's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0,
                "sparse count should be greater than 0, but now is ",
                sparse_count);
    // refine 固定 TND query + PA_BSND key: out [T, key.shape[2](恒1), refineCount]
    TORCH_CHECK(query_layout_str == "TND",
                "layout_query only supported TND for refine, but got ",
                query_layout_str);
    TORCH_CHECK(key_layout_str == "PA_BSND",
                "layout_key only supported PA_BSND for refine, but got ",
                key_layout_str);
    output_size = {query.size(DIM_0), key.size(DIM_2), sparse_count};

    return at::empty(output_size, query.options().dtype(at::kInt));
}

at::Tensor npu_indexer_refine(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& candidates,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const c10::optional<at::Tensor>& block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
    TORCH_CHECK(candidates.numel() > 0, "Tensor candidates is empty.");

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    at::Tensor sparse_indices_out = construct_indexer_refine_output_tensor(
        query, key, sparse_count, query_layout_str, key_layout_str);

    char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

    EXEC_NPU_CMD(aclnnIndexerRefine, query, key, weights, candidates,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 query_layout_ptr, key_layout_ptr, sparse_count,
                 sparse_indices_out);

    return sparse_indices_out;
}
}  // namespace vllm_ascend

#endif  // INDEXER_REFINE_TORCH_ADPT_H
