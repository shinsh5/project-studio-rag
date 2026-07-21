import unittest
import numpy as np

from entropy import calculate_pairwise_similarities, calculate_redundancy_entropy, calculate_diversity_entropy, compute_neighborhood_entropies
from indexer import segment_text, build_candidate_neighborhoods

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

if __name__ == '__main__':
    unittest.main()
