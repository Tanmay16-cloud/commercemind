def precision_at_k(relevant_items: set[str], retrieved_items: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not retrieved_items:
        return 0.0

    top_k = retrieved_items[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_items)
    return hits / len(top_k)


def recall_at_k(relevant_items: set[str], retrieved_items: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_items:
        return 0.0

    top_k = retrieved_items[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(relevant_items: set[str], retrieved_items: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = retrieved_items[:k]
    return float(any(item_id in relevant_items for item_id in top_k))


def reciprocal_rank_at_k(relevant_items: set[str], retrieved_items: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    for rank, item_id in enumerate(retrieved_items[:k], start=1):
        if item_id in relevant_items:
            return 1 / rank
    return 0.0
