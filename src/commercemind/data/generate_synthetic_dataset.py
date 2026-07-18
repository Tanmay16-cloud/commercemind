from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from commercemind.data.synthetic import (
    DEFAULT_SYNTHETIC_DATASET_NAME,
    SyntheticDatasetSpec,
    materialize_synthetic_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic CommerceMind dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_SYNTHETIC_DATASET_NAME,
        help="Dataset name under data/processed/.",
    )
    parser.add_argument(
        "--products",
        type=int,
        default=1000,
        help="Number of synthetic products to generate.",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=100,
        help="Number of synthetic users to generate interactions for.",
    )
    parser.add_argument(
        "--interactions",
        type=int,
        default=5000,
        help="Number of synthetic interaction events to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional processed dataset directory. Defaults to data/processed/<dataset>.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = materialize_synthetic_dataset(
            SyntheticDatasetSpec(
                dataset_name=args.dataset,
                product_count=args.products,
                user_count=args.users,
                interaction_count=args.interactions,
                seed=args.seed,
            ),
            processed_dir=args.output_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(
        "Generated synthetic dataset "
        f"'{result.dataset_name}' with "
        f"{result.product_count} products, "
        f"{result.user_count} users, "
        f"{result.interaction_count} interactions."
    )
    print(f"Products: {result.products_path}")
    print(f"Interactions: {result.interactions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
