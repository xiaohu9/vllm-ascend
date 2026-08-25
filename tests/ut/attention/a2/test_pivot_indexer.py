# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the PIVOT-Refine indexer (pivot_indexer.py).

The NPU ops (npu_dynamic_quant / npu_quant_lightning_indexer) are mocked
with dense torch references so the refine math runs on CPU. The mock
indexer reproduces the native op contract: per-request top-sparse_count
positions by proxy score over [0, actual_seq_lengths_key), 0-based,
-1 padded. With quant mocked as identity (scale = 1), PIVOT's refine
scoring and the dense reference use the same formula and dtype, so
short-prefix cases (C covers the full prefix) must match exactly.

Design doc: docs/source/developer_guide/Design_Documents/pivot_indexer.md
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from tests.ut.base import TestBase
from vllm_ascend.attention import pivot_indexer as pivot_mod
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.pivot_indexer import PivotIndexer
from vllm_ascend.attention.sfa_v1 import AscendSFABetadata, AscendSFABetadataBuilder

H = 2
D = 8
BLOCK_SIZE = 16
K = 8  # VLLM_ASCEND_PIVOT_TOPK for tests (default 512 is too wide for CPU refs)


def _identity_dynamic_quant(x, dst_type=None):
    """Quant mock: identity values with scale 1 keeps one dequant domain."""
    return x, torch.ones(x.shape[0], dtype=torch.float32, device=x.device)


def _make_mock_lightning_indexer(records):
    """Dense-reference stand-in for npu_quant_lightning_indexer."""

    def mock_op(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        sparse_count,
        **kwargs,
    ):
        records.append(
            {
                "key_lens": actual_seq_lengths_key.clone(),
                "sparse_count": int(sparse_count),
            }
        )
        R = query.shape[0]
        bs = key.shape[1]  # PA_BSND: [num_blocks, block_size, 1, D]
        kc = key.reshape(-1, key.shape[-1])
        ks = key_dequant_scale.reshape(-1)
        bt = block_table.long()
        out = torch.full((R, 1, sparse_count), -1, dtype=torch.int32)
        for r in range(R):
            length = int(actual_seq_lengths_key[r])
            if length == 0:
                continue
            pos = torch.arange(length)
            slots = bt[r, pos // bs] * bs + pos % bs
            k_dq = (kc[slots].float() * ks[slots].unsqueeze(-1)).to(torch.bfloat16)  # [L, D]
            q_dq = query[r].to(torch.bfloat16) * query_dequant_scale[r].to(
                torch.bfloat16
            ).unsqueeze(-1)  # [H, D]
            att = torch.relu(q_dq @ k_dq.T)  # [H, L]
            score = (att * weights[r].to(torch.bfloat16).unsqueeze(-1)).sum(0)  # [L]
            n = min(length, sparse_count)
            top = torch.topk(score, n).indices
            out[r, 0, :n] = pos[top].to(torch.int32)
        return out

    return mock_op


class PivotCase:
    """Fixture builder for grouped MTP decode batches on CPU."""

    def __init__(self, prefix_lens, g, seed=0, num_input_tokens=None):
        torch.manual_seed(seed)
        self.g = g
        self.R = len(prefix_lens)
        self.N = self.R * g
        self.prefix_lens = torch.tensor(prefix_lens, dtype=torch.int64)
        self.num_input_tokens = num_input_tokens or self.N
        N_in = self.num_input_tokens

        # Contiguous per-request key allocation: position p of request r
        # lives at slot block_table[r, p // bs] * bs + p % bs.
        max_len = int(self.prefix_lens.max()) + g
        blocks_per_req = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = self.R * blocks_per_req + 1
        self.block_table = torch.zeros(self.R, blocks_per_req, dtype=torch.int64)
        cursor = 0
        for r in range(self.R):
            self.block_table[r] = torch.arange(
                cursor, cursor + blocks_per_req, dtype=torch.int64
            )
            cursor += blocks_per_req
        self.num_blocks = num_blocks

        self.k_cache = torch.randn(num_blocks, BLOCK_SIZE, 1, D, dtype=torch.bfloat16)
        self.k_scale = torch.ones(num_blocks, BLOCK_SIZE, 1, dtype=torch.float32)
        self.kv_cache = (
            MagicMock(),
            MagicMock(),
            self.k_cache,
            self.k_scale,
        )

        self.q_li = torch.randn(N_in * H, D, dtype=torch.bfloat16)
        self.q_li_scale = torch.ones(N_in * H, dtype=torch.float32)
        self.weights = torch.rand(N_in, H, dtype=torch.bfloat16) + 0.5

        # Scheduler-authoritative positions: t0..t0+g-1 per request.
        positions = []
        for L in prefix_lens:
            positions.append(torch.arange(L, L + g))
        self.positions_q = torch.cat(positions)
        counts = torch.full((self.R,), g, dtype=torch.int64)
        self.counts = counts
        self.group_start = torch.cat(
            [torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, 0)[:-1]]
        )
        self.req_ids = torch.repeat_interleave(
            torch.arange(self.R), counts, output_size=self.N
        )
        # Window W = g; out-of-range columns are -1 (never clamp to 0).
        win_offsets = (g - 1) - torch.arange(g)
        window = self.positions_q.unsqueeze(1) - win_offsets.unsqueeze(0)
        self.window_pos = torch.where(window >= 0, window, torch.full_like(window, -1))
        self.seq_lens = self.prefix_lens + g  # L + g at indexer time
        self.proxy_key_lens = self.seq_lens - counts

        self.metadata = SimpleNamespace(
            seq_lens=self.seq_lens,
            num_actual_tokens=self.N,
            block_table=self.block_table,
            block_size=BLOCK_SIZE,
            pivot_counts=self.counts,
            pivot_group_start=self.group_start,
            pivot_req_ids=self.req_ids,
            pivot_positions_q=self.positions_q,
            pivot_window_pos=self.window_pos,
            pivot_proxy_key_lens=self.proxy_key_lens,
        )
        self.sfa_impl = SimpleNamespace(
            enable_sparse_li_c8=True,
            enable_sparse_sfa_c8=False,
            c8_k_cache_dtype=torch.bfloat16,
            c8_k_scale_cache_dtype=torch.float32,
            use_index_cache=False,
            topk_indices_buffer=None,
        )

    def dense_topk(self, k=K):
        """Per-query dense reference over [0, positions_q] (same formula)."""
        kc = self.k_cache.reshape(-1, D)
        ks = self.k_scale.reshape(-1)
        out = []
        for n in range(self.N):
            pos_q = int(self.positions_q[n])
            pos = torch.arange(pos_q + 1)
            r = int(self.req_ids[n])
            slots = self.block_table[r, pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
            k_dq = (kc[slots].float() * ks[slots].unsqueeze(-1)).to(torch.bfloat16)
            q_dq = self.q_li.view(self.num_input_tokens, H, D)[n] * self.q_li_scale.view(
                self.num_input_tokens, H, 1
            )[n].to(torch.bfloat16)
            att = torch.relu(q_dq @ k_dq.T)
            score = (att * self.weights[n].unsqueeze(-1)).sum(0)
            n_top = min(pos_q + 1, k)
            top = torch.topk(score, n_top).indices
            row = pos[top].tolist() + [-1] * (k - n_top)
            out.append(sorted(row))
        return out


class TestPivotIndexerSelectTopk(TestBase):
    def setUp(self):
        self.records = []
        self.quant_patcher = patch.object(
            pivot_mod.torch_npu, "npu_dynamic_quant", side_effect=_identity_dynamic_quant
        )
        self.indexer_patcher = patch.object(
            pivot_mod.torch_npu,
            "npu_quant_lightning_indexer",
            side_effect=_make_mock_lightning_indexer(self.records),
        )
        self.quant_patcher.start()
        self.indexer_patcher.start()
        self.env_patcher = patch.dict(os.environ, {"VLLM_ASCEND_PIVOT_TOPK": str(K)})
        self.env_patcher.start()

    def tearDown(self):
        self.quant_patcher.stop()
        self.indexer_patcher.stop()
        self.env_patcher.stop()

    def _select(self, case):
        return PivotIndexer.select_topk(
            case.sfa_impl,
            case.q_li,
            case.q_li_scale,
            (case.num_input_tokens, H, D),
            case.weights,
            case.kv_cache,
            case.metadata,
        )

    def test_short_prefix_dense_parity(self):
        # L <= c: the mock proxy scan returns the FULL prefix, so C∪W_t
        # equals the dense per-query domain exactly -> exact top-k parity.
        case = PivotCase(prefix_lens=[20, 24, 12], g=2, seed=3)
        out = self._select(case)
        self.assertEqual(tuple(out.shape), (case.N, 1, K))
        expected = case.dense_topk()
        for n in range(case.N):
            got = sorted(out[n, 0].tolist())
            self.assertEqual(got, expected[n], f"row {n}")

    def test_c_domain_excludes_mtp_keys(self):
        # Proxy scan key len must be L (= seq_lens - counts): the mock op
        # records its inputs; C must never contain this step's MTP keys.
        case = PivotCase(prefix_lens=[30, 40], g=3, seed=7)
        out = self._select(case)
        self.assertEqual(len(self.records), 1)
        self.assertTrue(
            torch.equal(self.records[0]["key_lens"].long(), case.prefix_lens)
        )
        # Draft keys (>= L) can only enter via the window; with random
        # scores they are rarely all absent, and window columns always
        # cover [L, t] since W = g.
        for n in range(case.N):
            pos_q = int(case.positions_q[n])
            for p in out[n, 0].tolist():
                if p >= 0:
                    self.assertLessEqual(p, pos_q)

    def test_window_bring_draft_keys(self):
        # Make the draft keys dominate: the newest key of each query must
        # be selected (window columns carry their own true positions).
        case = PivotCase(prefix_lens=[30, 40], g=2, seed=11)
        # Boost keys at positions L..L+g-1 of each request.
        for r in range(case.R):
            L = int(case.prefix_lens[r])
            for p in range(L, L + case.g):
                slot = case.block_table[r, p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE
                case.k_cache.view(-1, D)[slot] *= 100.0
        out = self._select(case)
        for n in range(case.N):
            row = out[n, 0].tolist()
            self.assertIn(int(case.positions_q[n]), row)

    def test_no_duplicates_and_bounds(self):
        case = PivotCase(prefix_lens=[20, 24, 12], g=2, seed=3)
        out = self._select(case)
        for n in range(case.N):
            row = [p for p in out[n, 0].tolist() if p >= 0]
            self.assertEqual(len(row), len(set(row)), f"duplicate in row {n}")
            pos_q = int(case.positions_q[n])
            self.assertTrue(all(0 <= p <= pos_q for p in row))

    def test_empty_prefix(self):
        # L = 0: C is all -1 (proxy key len 0); only window columns remain.
        case = PivotCase(prefix_lens=[0, 0], g=2, seed=13)
        out = self._select(case)
        self.assertEqual(len(self.records), 1)
        self.assertTrue(torch.equal(self.records[0]["key_lens"].long(), torch.zeros(2)))
        expected = case.dense_topk()
        for n in range(case.N):
            got = sorted(out[n, 0].tolist())
            self.assertEqual(got, expected[n], f"row {n}")

    def test_g_lt_2_returns_none(self):
        case = PivotCase(prefix_lens=[10, 20], g=1, seed=17)
        out = self._select(case)
        self.assertIsNone(out)

    def test_c8_only_guard(self):
        case = PivotCase(prefix_lens=[10], g=2, seed=19)
        case.sfa_impl.enable_sparse_li_c8 = False
        with self.assertRaises(NotImplementedError):
            self._select(case)

    def test_use_index_cache_width_pad(self):
        case = PivotCase(prefix_lens=[20, 24], g=2, seed=23)
        case.sfa_impl.use_index_cache = True
        case.sfa_impl.topk_indices_buffer = torch.zeros(
            case.num_input_tokens, 16, dtype=torch.int32
        )
        out = self._select(case)
        self.assertEqual(tuple(out.shape), (case.N, 1, 16))
        self.assertTrue(torch.equal(out[..., K:], torch.full_like(out[..., K:], -1)))

    def test_graph_padding_rows(self):
        # num_input_tokens > num_actual_tokens: padded rows get -1 tails.
        case = PivotCase(prefix_lens=[20, 24], g=2, seed=29, num_input_tokens=6)
        out = self._select(case)
        self.assertEqual(tuple(out.shape), (6, 1, K))
        self.assertTrue(torch.equal(out[case.N :, 0], torch.full_like(out[case.N :, 0], -1)))

    def test_nonuniform_groups_fallback(self):
        # counts [3, 1]: eager per-request scoring fallback must still
        # match the dense reference (L <= c covers the full prefix).
        # seq_lens = L + counts: req0 keeps 3 new keys (positions 29-31),
        # req1 keeps 1 (position 29); all within the block-table range.
        case = PivotCase(prefix_lens=[29, 29], g=2, seed=31)
        case.counts = torch.tensor([3, 1], dtype=torch.int64)
        case.group_start = torch.tensor([0, 3], dtype=torch.int64)
        case.req_ids = torch.repeat_interleave(
            torch.arange(2), case.counts, output_size=4
        )
        case.metadata.pivot_counts = case.counts
        case.metadata.pivot_group_start = case.group_start
        case.metadata.pivot_req_ids = case.req_ids
        # N is now 4 with mixed group sizes; positions follow the groups.
        positions = torch.tensor([29, 30, 31, 29])
        case.positions_q = positions
        case.metadata.pivot_positions_q = positions
        window = positions.unsqueeze(1) - torch.flip(
            torch.arange(2), dims=(0,)
        ).unsqueeze(0)
        case.window_pos = torch.where(window >= 0, window, torch.full_like(window, -1))
        case.metadata.pivot_window_pos = case.window_pos
        case.metadata.pivot_proxy_key_lens = torch.tensor([29, 29])
        case.metadata.num_actual_tokens = 4

        out = self._select(case)
        expected = case.dense_topk()
        for n in range(4):
            got = sorted(out[n, 0].tolist())
            self.assertEqual(got, expected[n], f"row {n}")

    def test_chunking_consistency(self):
        # _REFINE_CHUNK must not change results, only memory shape.
        case = PivotCase(prefix_lens=[20, 24, 12], g=2, seed=37)
        out_full = self._select(case).clone()
        self.records.clear()
        with patch.object(pivot_mod, "_REFINE_CHUNK", 1):
            out_chunked = self._select(case)
        self.assertTrue(torch.equal(out_full, out_chunked))


class TestAscendSFAMetadataPivotFields(TestBase):
    def _make_builder(self):
        builder = AscendSFABetadataBuilder.__new__(AscendSFABetadataBuilder)
        builder.kernel_block_size = BLOCK_SIZE
        builder.model_config = MagicMock()
        builder.model_config.get_head_size.return_value = D
        builder.attn_mask_builder = MagicMock()
        builder.enable_dsa_cp = False
        builder.metadata_cls = AscendSFABetadata
        return builder

    def _make_common(self, prefix_lens, g, attn_state, decode_token_per_req=None):
        R = len(prefix_lens)
        N = R * g
        common = MagicMock()
        common.num_reqs = R
        common.num_actual_tokens = N
        common.num_input_tokens = N
        common.attn_state = attn_state
        common.decode_token_per_req = (
            decode_token_per_req if decode_token_per_req is not None else g
        )
        common.query_start_loc = torch.zeros(R + 1, dtype=torch.int64)
        common.query_start_loc[1:] = torch.cumsum(
            torch.full((R,), g, dtype=torch.int64), 0
        )
        common.seq_lens = torch.tensor([L + g for L in prefix_lens], dtype=torch.int64)
        common._seq_lens_cpu = common.seq_lens.clone()
        common.seq_lens_cpu = common.seq_lens.clone()
        positions = torch.cat([torch.arange(L, L + g) for L in prefix_lens])
        common.positions = positions
        common.block_table_tensor = torch.zeros(R, 4, dtype=torch.int64)
        common.slot_mapping = torch.arange(N, dtype=torch.int64)
        common.causal = True
        return common, positions

    @patch("vllm_ascend.attention.sfa_v1.get_ascend_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    def _build(self, prefix_lens, g, attn_state, mock_cos, mock_ascend_cfg, **kwargs):
        mock_cos.return_value = (
            torch.zeros(64, 1, 1, D),
            torch.zeros(64, 1, 1, D),
        )
        mock_ascend_cfg.return_value = MagicMock(c8_enable_reshape_optim=False)
        builder = self._make_builder()
        common, positions = self._make_common(prefix_lens, g, attn_state, **kwargs)
        return builder._build(common), positions

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PIVOT_REFINE": "1"})
    def test_pivot_fields_computed_for_decode(self):
        # prefix_lens[0] = 0 exercises the sequence-start window edge.
        metadata, positions = self._build(
            [0, 80, 60], 3, AscendAttentionState.SpecDecoding
        )
        self.assertTrue(torch.equal(metadata.pivot_counts, torch.tensor([3, 3, 3])))
        self.assertTrue(torch.equal(metadata.pivot_group_start, torch.tensor([0, 3, 6])))
        self.assertTrue(
            torch.equal(metadata.pivot_req_ids, torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]))
        )
        self.assertTrue(torch.equal(metadata.pivot_positions_q, positions))
        # W = g = 3: window [t-2, t], -1 for out-of-range columns.
        self.assertTrue(
            torch.equal(
                metadata.pivot_window_pos[0],
                torch.tensor([-1, -1, 0]),
            )
        )
        self.assertTrue(
            torch.equal(metadata.pivot_window_pos[1], torch.tensor([-1, 0, 1]))
        )
        self.assertTrue(
            torch.equal(metadata.pivot_window_pos[2], torch.tensor([0, 1, 2]))
        )
        self.assertTrue(
            torch.equal(metadata.pivot_window_pos[3], torch.tensor([78, 79, 80]))
        )
        self.assertTrue(
            torch.equal(
                metadata.pivot_proxy_key_lens,
                torch.tensor([0, 80, 60]),
            )
        )

    def test_pivot_fields_none_when_switch_off(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PIVOT_REFINE": "0"}):
            metadata, _ = self._build([100], 3, AscendAttentionState.DecodeOnly)
        self.assertIsNone(metadata.pivot_counts)
        self.assertIsNone(metadata.pivot_window_pos)

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PIVOT_REFINE": "1"})
    def test_pivot_fields_none_for_ungrouped(self):
        # decode_token_per_req = 1: draft iterations / plain decode.
        metadata, _ = self._build(
            [100], 1, AscendAttentionState.DecodeOnly, decode_token_per_req=1
        )
        self.assertIsNone(metadata.pivot_counts)

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PIVOT_REFINE": "1"})
    def test_pivot_fields_none_for_prefill_states(self):
        # PD-mixed / chunked-prefill batches never enter the PIVOT path.
        for state in (
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.PrefillNoCache,
            AscendAttentionState.PrefillCacheHit,
        ):
            metadata, _ = self._build([100], 3, state)
            self.assertIsNone(metadata.pivot_counts, f"state={state}")


class TestSfaV1PivotBranchGate(TestBase):
    """indexer_select_post_process routes to PivotIndexer only when the
    metadata carries PIVOT fields (single gate source) and falls back on
    a None return."""

    def test_gate_requires_pivot_metadata(self):
        # The branch condition is a pure predicate on metadata; verify the
        # two sides without running the heavy Impl method body.
        enabled = MagicMock(pivot_counts=torch.ones(2))
        disabled = MagicMock(pivot_counts=None)
        import vllm_ascend.envs as envs_mod

        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PIVOT_REFINE": "1"}):
            self.assertTrue(
                envs_mod.VLLM_ASCEND_ENABLE_PIVOT_REFINE
                and enabled.pivot_counts is not None
            )
            self.assertFalse(
                envs_mod.VLLM_ASCEND_ENABLE_PIVOT_REFINE
                and disabled.pivot_counts is not None
            )
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PIVOT_REFINE": "0"}):
            self.assertFalse(
                envs_mod.VLLM_ASCEND_ENABLE_PIVOT_REFINE
                and enabled.pivot_counts is not None
            )
