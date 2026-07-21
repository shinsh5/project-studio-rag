"""
ROI-RAG 엔트로피 연산 모듈.
Pairwise Cosine Similarity 행렬 연산, 중복 엔트로피(RE), 다양성 엔트로피(DE) 계산을 수행합니다.
"""
import numpy as np

def calculate_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity matrix for a set of embeddings.
    Assumes embeddings are of shape (m, d).
    """
    m = embeddings.shape[0]
    if m == 0:
        return np.empty((0, 0), dtype=np.float32)
    
    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Clamp similarities to [0, 1] for probability mapping
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
    return similarity_matrix

def calculate_redundancy_entropy(similarity_matrix: np.ndarray) -> float:
    """
    RE (Redundancy Entropy): Measures similarity distribution concentration in the neighborhood.
    High RE indicates high redundancy (similarities across neighbors are uniform and high).
    """
    m = similarity_matrix.shape[0]
    if m <= 1:
        return 0.0
    
    row_entropies = []
    for i in range(m):
        row = similarity_matrix[i]
        row_sum = np.sum(row)
        if row_sum == 0:
            row_entropies.append(0.0)
            continue
            
        prob_dist = row / row_sum
        # Compute Shannon entropy in bits (log2)
        # Add epsilon to prevent log(0)
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-12))
        # Normalize by max possible entropy log2(m)
        entropy = entropy / np.log2(m)
        row_entropies.append(entropy)
        
    return float(np.mean(row_entropies))

def calculate_diversity_entropy(similarity_matrix: np.ndarray) -> float:
    """
    DE (Diversity Entropy): Captures the dispersion of semantic content.
    Higher DE indicates that candidate segments provide more complementary diversity.
    """
    m = similarity_matrix.shape[0]
    if m <= 1:
        return 0.0
    
    # Compute distance matrix (D = 1 - S)
    distance_matrix = 1.0 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
    
    row_entropies = []
    for i in range(m):
        row = distance_matrix[i]
        row_sum = np.sum(row)
        if row_sum == 0:
            row_entropies.append(0.0)
            continue
            
        prob_dist = row / row_sum
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-12))
        entropy = entropy / np.log2(m)
        row_entropies.append(entropy)
        
    return float(np.mean(row_entropies))

def compute_neighborhood_entropies(embeddings: np.ndarray) -> tuple[float, float]:
    """
    Convenience function to compute both RE and DE from a set of embedding vectors.
    Returns: (re, de)
    """
    if embeddings.shape[0] <= 1:
        return 0.0, 0.0
    S = calculate_pairwise_similarities(embeddings)
    re = calculate_redundancy_entropy(S)
    de = calculate_diversity_entropy(S)
    return re, de
