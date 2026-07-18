from commercemind.evaluation.benchmarks import build_benchmark_dataset


def test_sample_benchmark_contains_products_interactions_and_labels() -> None:
    benchmark = build_benchmark_dataset("sample")

    assert benchmark.name == "sample"
    assert benchmark.products.height == 12
    assert benchmark.interactions.height == 8
    assert len(benchmark.search_cases) == 6
    assert len(benchmark.recommendation_cases) == 4


def test_recommendation_labels_are_not_already_seen_by_users() -> None:
    benchmark = build_benchmark_dataset("sample")
    seen_by_user: dict[str, set[str]] = {}

    for row in benchmark.interactions.iter_rows(named=True):
        seen_by_user.setdefault(row["user_id"], set()).add(row["item_id"])

    for case in benchmark.recommendation_cases:
        assert case.relevant_item_ids.isdisjoint(seen_by_user.get(case.user_id, set()))
