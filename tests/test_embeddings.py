import numpy as np

from commercemind.retrieval.embeddings import HashingTextEmbedder, cosine_similarity


def test_hashing_text_embedder_returns_normalized_vectors() -> None:
    embedder = HashingTextEmbedder(dimensions=32)

    vectors = embedder.encode(["running shoes", "office shirt"])

    assert vectors.shape == (2, 32)
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_hashing_text_embedder_is_stable_across_instances() -> None:
    first_embedder = HashingTextEmbedder(dimensions=32)
    second_embedder = HashingTextEmbedder(dimensions=32)

    first_vector = first_embedder.encode(["running shoes"])
    second_vector = second_embedder.encode(["running shoes"])

    assert np.array_equal(first_vector, second_vector)


def test_cosine_similarity_scores_closest_vector_highest() -> None:
    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    document_vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scores = cosine_similarity(query_vector, document_vectors)

    assert scores[0] > scores[1]
    assert scores[0] == 1.0
