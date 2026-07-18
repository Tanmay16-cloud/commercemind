from pathlib import Path

from commercemind.ranking.learned import load_learned_ranking_model
from commercemind.ranking.training import main, train_ranker_from_benchmark


def test_train_ranker_from_benchmark_writes_model_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "ranker.json"

    result = train_ranker_from_benchmark("sample", output_path=output_path)
    model = load_learned_ranking_model(output_path)

    assert result.benchmark_name == "sample"
    assert result.model_path == output_path
    assert result.training_examples > 0
    assert result.positive_examples > 0
    assert model.training_examples == result.training_examples


def test_train_ranker_cli_writes_model_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "ranker.json"

    exit_code = main(["--benchmark", "demo", "--output-path", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
