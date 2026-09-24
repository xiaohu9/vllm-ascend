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
    const at::Tensor& q_bar, int64_t coarse_count, int64_t has_window, int64_t group_size)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < q_bar.sizes().size(); i++) {
        TORCH_CHECK(q_bar.size(i) > 0,
                    "All values within q_bar's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", q_bar.size(i));
    }
    TORCH_CHECK(coarse_count > 0,
                "coarse count should be greater than 0, but now is ",
                coarse_count);
    // 固定 TND q_bar + PA_BSND key:候选 [R, W'],W' = coarse + (has_window ? 2g-1 : 0)
    const int64_t req_num = q_bar.size(DIM_0);
    int64_t out_w = coarse_count;
    if (has_window != 0) {
        out_w = coarse_count + 2 * group_size - 1;
    }
    output_size = {req_num, out_w};

    at::Tensor candidates_out = at::empty(output_size, q_bar.options().dtype(at::kInt));
    // at::zeros(非 empty):全零批(所有行粗筛域空)走 kernel ProcessInvalid
    // 清理路径,该路径的 aslk 写实测不可靠(同核多发 init 部分丢失),aslk'
    // 语义恰为 0 → host 保证;普通路径 kernel 必然覆盖写 aslk',此零初始化无感。
    at::Tensor aslk_out = at::zeros({req_num}, q_bar.options().dtype(at::kInt));
    return std::tuple<at::Tensor, at::Tensor>(candidates_out, aslk_out);
}

std::tuple<at::Tensor, at::Tensor> npu_indexer_coarse_screen(
    const at::Tensor& q_bar, const at::Tensor& w_bar,
    const at::Tensor& key,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const at::Tensor& block_table, int64_t coarse_count, int64_t has_window,
    int64_t group_size)
{
    // 图安全:仅做 shape/dtype 存在性检查,捕获期不读张量数据
    TORCH_CHECK(q_bar.numel() > 0, "Tensor q_bar is empty.");
    TORCH_CHECK(w_bar.numel() > 0, "Tensor w_bar is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(block_table.numel() > 0, "Tensor block_table is empty.");

    auto coarse_screen_output = construct_indexer_coarse_screen_output_tensor(
        q_bar, coarse_count, has_window, group_size);
    at::Tensor candidates_out = std::get<0>(coarse_screen_output);
    at::Tensor aslk_out = std::get<1>(coarse_screen_output);

    EXEC_NPU_CMD(aclnnIndexerCoarseScreen, q_bar, w_bar, key,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 coarse_count, has_window, group_size, candidates_out, aslk_out);

    return std::tuple<at::Tensor, at::Tensor>(candidates_out, aslk_out);
}
}  // namespace vllm_ascend

#endif  // INDEXER_COARSE_SCREEN_TORCH_ADPT_H
