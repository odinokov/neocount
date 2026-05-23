#!/usr/bin/env python3
"""Build normalized neocount feature tables from TSV reports.

Normalization strategy:
  norm_reads = read pairs in control regions (chr2:95M-242M, chr19:0-24.2M)
  value = pair_count * 1_000_000 / norm_reads

This is region-based depth normalization anchored to stable genomic regions, not
global total-reads normalization. The 1_000_000 factor converts the small ratio
(pair_count / norm_reads) to readable integer scale.

For downstream ML tasks (classification, clustering): Consider applying per-sample
min-max normalization to remove remaining per-sample biases before PCA or other
dimensionality reduction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm


AF_BINS = tuple(range(6))
KEY_COLUMNS = ["sample", "bam", "db", "k"]
INPUT_COLUMNS = [
    "sample",
    "bam",
    "db",
    "k",
    "organ",
    "variant_class",
    "af_bin",
    "pair_count",
    "norm_reads",
]
SCALE = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize one-level neocount *.tsv reports into a harmonized "
            "CSV feature matrix."
        )
    )
    parser.add_argument("path", type=Path, help="Directory containing *.tsv files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    return parser.parse_args()


def discover_tsv_files(path: Path) -> list[Path]:
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"input path must be a directory containing *.tsv files: {path}")

    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".tsv")
    if not files:
        raise SystemExit(f"no *.tsv files found in {path}")
    return files


def read_needed_columns(tsv_path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(
            tsv_path,
            separator="\t",
            columns=INPUT_COLUMNS,
            schema_overrides={
                "sample": pl.String,
                "bam": pl.String,
                "db": pl.String,
                "k": pl.Int64,
                "organ": pl.String,
                "variant_class": pl.String,
                "af_bin": pl.Int8,
                "pair_count": pl.Int64,
                "norm_reads": pl.Int64,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"failed to read required columns from {tsv_path}: {exc}") from exc


def validate_frame(df: pl.DataFrame, tsv_path: Path) -> None:
    null_required = df.select(
        pl.any_horizontal(
            pl.col("sample").is_null(),
            pl.col("bam").is_null(),
            pl.col("db").is_null(),
            pl.col("k").is_null(),
            pl.col("organ").is_null(),
            pl.col("variant_class").is_null(),
            pl.col("af_bin").is_null(),
            pl.col("pair_count").is_null(),
            pl.col("norm_reads").is_null(),
        ).sum()
    ).item()
    if null_required:
        raise ValueError(f"{tsv_path} has missing required values")

    bad_bins = (
        df.filter(~pl.col("af_bin").is_in(AF_BINS))
        .select("af_bin")
        .unique()
    )
    if bad_bins.height:
        values = ", ".join(str(v) for v in bad_bins.get_column("af_bin").to_list())
        raise ValueError(f"{tsv_path} has unsupported af_bin value(s): {values}")

    if df.filter(pl.col("pair_count") < 0).height:
        raise ValueError(f"{tsv_path} has negative pair_count values")

    if df.filter(pl.col("norm_reads") <= 0).height:
        raise ValueError(f"{tsv_path} has nonpositive norm_reads values")

    norm_conflicts = (
        df.group_by(KEY_COLUMNS)
        .agg(pl.col("norm_reads").n_unique().alias("norm_reads_values"))
        .filter(pl.col("norm_reads_values") > 1)
    )
    if norm_conflicts.height:
        raise ValueError(f"{tsv_path} has conflicting norm_reads within sample/bam/db/k")


def extend_ordered(values: list[str], seen: set[str], new_values: list[str]) -> None:
    for value in new_values:
        if value not in seen:
            seen.add(value)
            values.append(value)


def summarize_dimension(df: pl.DataFrame, dimension: str) -> pl.DataFrame:
    return (
        df.with_columns(
            (
                pl.col(dimension)
                + pl.lit("_")
                + pl.col("af_bin").cast(pl.String)
            ).alias("feature")
        )
        .group_by(KEY_COLUMNS + ["norm_reads", "feature"])
        .agg(pl.col("pair_count").sum().alias("pair_count"))
    )


def summarize_file(tsv_path: Path) -> tuple[pl.DataFrame, list[str], list[str]]:
    df = read_needed_columns(tsv_path)
    validate_frame(df, tsv_path)

    organs = df.get_column("organ").unique(maintain_order=True).to_list()
    variant_classes = df.get_column("variant_class").unique(maintain_order=True).to_list()

    summary = pl.concat(
        [
            summarize_dimension(df, "organ"),
            summarize_dimension(df, "variant_class"),
        ],
        how="vertical",
    )
    return summary, organs, variant_classes


def build_feature_order(organs: list[str], variant_classes: list[str]) -> list[str]:
    organ_features = [f"{organ}_{af_bin}" for organ in organs for af_bin in AF_BINS]
    variant_features = [
        f"{variant_class}_{af_bin}"
        for variant_class in variant_classes
        for af_bin in AF_BINS
    ]

    all_features = organ_features + variant_features
    duplicate_count = len(all_features) - len(set(all_features))
    if duplicate_count:
        raise SystemExit(
            "feature-name collision between organ and variant_class labels; "
            "single-underscore names are ambiguous"
        )
    return all_features


def validate_global_norm_reads(summary: pl.DataFrame) -> None:
    conflicts = (
        summary.group_by(KEY_COLUMNS)
        .agg(pl.col("norm_reads").n_unique().alias("norm_reads_values"))
        .filter(pl.col("norm_reads_values") > 1)
    )
    if conflicts.height:
        raise SystemExit("conflicting norm_reads across files for at least one sample/bam/db/k")


def build_wide_table(summary: pl.DataFrame, feature_order: list[str]) -> pl.DataFrame:
    validate_global_norm_reads(summary)

    norms = summary.group_by(KEY_COLUMNS).agg(pl.col("norm_reads").first())
    normalized = (
        summary.group_by(KEY_COLUMNS + ["feature"])
        .agg(pl.col("pair_count").sum().alias("pair_count"))
        .join(norms, on=KEY_COLUMNS, how="left")
        .with_columns(
            (
                pl.col("pair_count").cast(pl.Float64)
                * pl.lit(SCALE)
                / pl.col("norm_reads").cast(pl.Float64)
            ).alias("value")
        )
        .select(KEY_COLUMNS + ["feature", "value"])
    )

    wide = normalized.pivot(
        on="feature",
        index=KEY_COLUMNS,
        values="value",
        aggregate_function="first",
    )

    missing_features = [name for name in feature_order if name not in wide.columns]
    if missing_features:
        wide = wide.with_columns(pl.lit(0.0).alias(name) for name in missing_features)

    return wide.select(
        KEY_COLUMNS
        + [pl.col(name).fill_null(0.0).alias(name) for name in feature_order]
    ).sort(KEY_COLUMNS)


def main() -> int:
    args = parse_args()
    files = discover_tsv_files(args.path)

    summaries: list[pl.DataFrame] = []
    organs: list[str] = []
    variant_classes: list[str] = []
    seen_organs: set[str] = set()
    seen_variant_classes: set[str] = set()

    for tsv_path in tqdm(files, desc="TSV files", unit="file"):
        summary, file_organs, file_variant_classes = summarize_file(tsv_path)
        summaries.append(summary)
        extend_ordered(organs, seen_organs, file_organs)
        extend_ordered(variant_classes, seen_variant_classes, file_variant_classes)

    feature_order = build_feature_order(organs, variant_classes)
    combined = pl.concat(summaries, how="vertical")
    wide = build_wide_table(combined, feature_order)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wide.write_csv(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
