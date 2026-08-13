import unittest
import numpy as np
from unittest.mock import patch

from entropy import calculate_pairwise_similarities, calculate_redundancy_entropy, calculate_diversity_entropy, compute_neighborhood_entropies
from indexer import segment_text, build_candidate_neighborhoods
import config
import roi_rag

class TestROIRAGCore(unittest.TestCase):
    
    def test_segment_text(self):
        text = "word " * 500
        # Config has CHUNK_SIZE = 200, CHUNK_OVERLAP = 50
        chunks = segment_text(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0].split()), 200)
        self.assertEqual(len(chunks[1].split()), 200)
        
    def test_entropy_identical_vectors(self):
        # 3 identical unit vectors (highly redundant)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        S = calculate_pairwise_similarities(embeddings)
        re = calculate_redundancy_entropy(S)
        de = calculate_diversity_entropy(S)
        
        # When identical, S is all ones
        # Normalized similarity rows are [1/3, 1/3, 1/3], Shannon entropy is log2(3).
        # Since calculate_redundancy_entropy normalizes by log2(m), re should equal 1.0.
        self.assertAlmostEqual(re, 1.0, places=3)
        # Distances are all 0, DE is 0
        self.assertAlmostEqual(de, 0.0, places=3)
        
    def test_entropy_orthogonal_vectors(self):
        # 3 orthogonal vectors (perfectly diverse / no redundancy)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        S = calculate_pairwise_similarities(embeddings)
        re = calculate_redundancy_entropy(S)
        de = calculate_diversity_entropy(S)
        
        # S is identity matrix
        # Normalized row probabilities are [1.0, 0, 0], row entropy is 0.0
        self.assertAlmostEqual(re, 0.0, places=3)
        # Distance matrix has zeros on diagonal, ones elsewhere: [0, 1, 1] -> normalized to [0, 0.5, 0.5]
        # Shannon entropy = - 2*(0.5 * log2(0.5)) = 1.0. Divided by log2(3) = 1.0 / log2(3) = 0.63093.
        self.assertAlmostEqual(de, 1.0 / np.log2(3), places=3)
        
    def test_candidate_neighborhoods(self):
        embeddings = np.array([
            [1.0, 0.0], # A
            [0.9, 0.1], # B (similar to A)
            [0.0, 1.0], # C (orthogonal to A)
        ], dtype=np.float32)
        
        neighborhoods, sim_matrix = build_candidate_neighborhoods(embeddings, k=2)
        self.assertEqual(len(neighborhoods), 3)
        
        # Neighborhood of A (index 0) should contain A (0) and B (1)
        self.assertIn(0, neighborhoods[0])
        self.assertIn(1, neighborhoods[0])
        self.assertNotIn(2, neighborhoods[0])
        self.assertEqual(sim_matrix.shape, (3, 3))

class TestParentDeduplication(unittest.TestCase):
    def test_deduplicates_parents_across_retrieved_eus(self):
        seen = set()
        first, first_duplicates = roi_rag._deduplicate_parent_ids([2, 3], seen)
        second, second_duplicates = roi_rag._deduplicate_parent_ids([3, 4, 4], seen)

        self.assertEqual(first, [2, 3])
        self.assertEqual(first_duplicates, 0)
        self.assertEqual(second, [4])
        self.assertEqual(second_duplicates, 2)
        self.assertEqual(seen, {2, 3, 4})


class TestStbRetrievalModes(unittest.TestCase):
    """Small-to-Big offers two query-time context strategies over the same index."""

    def setUp(self):
        # 2 parents x 3 leaves; the EU hits one leaf in each parent.
        self.segments = [f"leaf-{i}" for i in range(6)]
        self.parent_chunks = ["leaf-0 leaf-1 leaf-2", "leaf-3 leaf-4 leaf-5"]
        self.leaf_to_parent = [0, 0, 0, 1, 1, 1]
        self.eu = {"eu_id": 0, "segment_indices": [4, 0], "re": 0.5, "de": 0.5,
                   "regime": "HIGH", "summary": "s"}

    @staticmethod
    def _no_trim(text):
        """BM25 선별을 끈 상태. 이 테스트는 컨텍스트 구성만 본다."""
        return text

    def _automerge(self, seen=None):
        body, _candidates, _duplicates = roi_rag._stb_automerge_context(
            self.eu, self.segments, self.parent_chunks, self.leaf_to_parent,
            self._no_trim, set() if seen is None else seen,
        )
        return body

    def _all_segments(self, eu=None):
        return roi_rag._stb_all_segments_context(
            eu or self.eu, self.segments, self._no_trim
        )

    def test_automerge_threshold_zero_keeps_every_touched_parent(self):
        with patch.object(config, "AUTOMERGE_THRESHOLD", 0.0):
            ctx = self._automerge()
        self.assertIn("[Parent Chunk #0]", ctx)
        self.assertIn("[Parent Chunk #1]", ctx)
        # Parent expansion pulls in neighbouring leaves not present in the EU.
        self.assertIn("leaf-2", ctx)

    def test_automerge_threshold_falls_back_to_leaves_when_all_parents_excluded(self):
        # Each parent holds 1 of 2 leaves (0.5), so 0.9 excludes both.
        with patch.object(config, "AUTOMERGE_THRESHOLD", 0.9):
            ctx = self._automerge()
        self.assertNotIn("[Parent Chunk", ctx)
        self.assertIn("Top Original Snippets", ctx)
        self.assertNotIn("leaf-2", ctx)

    def test_all_segments_sends_every_eu_leaf_in_document_order(self):
        ctx = self._all_segments()
        self.assertIn("All EU Segments", ctx)
        self.assertNotIn("[Parent Chunk", ctx)
        # Stored order is [4, 0]; output must be sorted to read continuously.
        self.assertLess(ctx.index("leaf-0"), ctx.index("leaf-4"))
        # Only the EU's own leaves, no parent neighbours.
        self.assertNotIn("leaf-2", ctx)

    def test_all_segments_is_smaller_than_automerge_expansion(self):
        with patch.object(config, "AUTOMERGE_THRESHOLD", 0.0):
            merged = self._automerge()
        self.assertLess(len(self._all_segments()), len(merged))

    def test_automerge_emits_a_shared_parent_only_once(self):
        seen = set()
        with patch.object(config, "AUTOMERGE_THRESHOLD", 0.0):
            first = self._automerge(seen)
            second = self._automerge(seen)
        self.assertIn("[Parent Chunk #0]", first)
        self.assertNotIn("[Parent Chunk", second)
        self.assertIn("already included above", second)

    def test_resolve_mode_defaults_and_rejects_unknown(self):
        with patch.object(config, "STB_RETRIEVAL_MODE", "all_segments"):
            self.assertEqual(roi_rag.resolve_stb_retrieval_mode(None), "all_segments")
            self.assertEqual(
                roi_rag.resolve_stb_retrieval_mode("automerge"), "automerge"
            )
        self.assertEqual(roi_rag.resolve_stb_retrieval_mode("bogus"), "automerge")

    def test_out_of_range_segment_indices_are_ignored(self):
        eu = {**self.eu, "segment_indices": [0, 99]}
        self.assertNotIn("99", self._all_segments(eu))

class TestLlamaResponseCache(unittest.TestCase):
    def setUp(self):
        roi_rag.clear_response_cache()

    def tearDown(self):
        roi_rag.clear_response_cache()

    def test_cache_key_tracks_prompt_model_options_and_index_version(self):
        first = roi_rag._response_cache_key("prompt", "index-v1")
        same = roi_rag._response_cache_key("prompt", "index-v1")
        changed_index = roi_rag._response_cache_key("prompt", "index-v2")
        with patch.object(config, "OLLAMA_SEED", config.OLLAMA_SEED + 1):
            changed_option = roi_rag._response_cache_key("prompt", "index-v1")

        self.assertEqual(first, same)
        self.assertNotEqual(first, changed_index)
        self.assertNotEqual(first, changed_option)

    def test_response_cache_is_lru_bounded(self):
        with patch.object(config, "LLM_RESPONSE_CACHE_MAX_SIZE", 2):
            roi_rag._store_cached_response("a", "answer-a")
            roi_rag._store_cached_response("b", "answer-b")
            self.assertEqual(roi_rag._get_cached_response("a"), "answer-a")
            roi_rag._store_cached_response("c", "answer-c")

        self.assertIsNone(roi_rag._get_cached_response("b"))
        self.assertEqual(roi_rag._get_cached_response("a"), "answer-a")
        self.assertEqual(roi_rag._get_cached_response("c"), "answer-c")

if __name__ == '__main__':
    unittest.main()
