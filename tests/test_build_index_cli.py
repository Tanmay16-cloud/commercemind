from pathlib import Path

from commercemind.retrieval.build_index import build_vector_index_from_catalog, main
from commercemind.retrieval.vector_index import INDEX_FILENAME, METADATA_FILENAME


def test_build_vector_index_from_sample_catalog(tmp_path: Path) -> None:
    result = build_vector_index_from_catalog("sample", output_dir=tmp_path)

    assert result.source == "sample"
    assert result.vector_count == 4
    assert result.dimensions == 384
    assert (tmp_path / INDEX_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()


def test_build_index_cli_writes_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "products"

    exit_code = main(["--source", "sample", "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / INDEX_FILENAME).exists()
    assert (output_dir / METADATA_FILENAME).exists()
