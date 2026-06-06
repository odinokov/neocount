"""Regression tests for helper/summarize_neocount.py: validation and wide-table math."""
import polars as pl
import pytest

import summarize_neocount as sm


# ---- small pure helpers ----------------------------------------------------

def test_extend_ordered():
    values, seen = [], set()
    sm.extend_ordered(values, seen, ["a", "b", "a"])
    sm.extend_ordered(values, seen, ["b", "c"])
    assert values == ["a", "b", "c"]


def test_build_feature_order():
    order = sm.build_feature_order(["Lung"], ["Missense"])
    # 6 af bins per label
    assert order == [f"Lung_{i}" for i in range(6)] + [f"Missense_{i}" for i in range(6)]


def test_build_feature_order_collision():
    # An organ label equal to a variant_class label collides on single-underscore names
    with pytest.raises(SystemExit):
        sm.build_feature_order(["X"], ["X"])


# ---- I/O + validation + math ----------------------------------------------

_HEADER = [
    "sample", "bam", "db", "k", "cancer_type", "organ", "variant_class",
    "af_bin", "af_bin_label", "pair_count", "norm_region", "norm_reads", "neomers",
]


def _write_tsv(path, rows):
    lines = ["\t".join(_HEADER)]
    for r in rows:
        lines.append("\t".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n")


def _row(organ="Lung", vc="Missense", af_bin=2, pair_count=10, norm_reads=1_000_000):
    return ["S1", "S1.bam", "db.ndb", 11, "LUAD", organ, vc,
            af_bin, "label", pair_count, "chr2q+chr19p", norm_reads, ""]


def test_summarize_pipeline_value_math(tmp_path):
    tsv = tmp_path / "S1.tsv"
    _write_tsv(tsv, [_row(organ="Lung", vc="Missense", af_bin=2,
                          pair_count=10, norm_reads=1_000_000)])
    summary, organs, vcs = sm.summarize_file(tsv)
    order = sm.build_feature_order(organs, vcs)
    wide = sm.build_wide_table(summary, order)
    # value = pair_count * 1e6 / norm_reads = 10 * 1e6 / 1e6 = 10.0
    assert wide["Lung_2"][0] == pytest.approx(10.0)
    assert wide["Missense_2"][0] == pytest.approx(10.0)
    # an unfilled feature is zero
    assert wide["Lung_0"][0] == pytest.approx(0.0)


def test_validate_frame_rejects_bad_af_bin(tmp_path):
    tsv = tmp_path / "bad.tsv"
    _write_tsv(tsv, [_row(af_bin=9)])
    df = sm.read_needed_columns(tsv)
    with pytest.raises(ValueError):
        sm.validate_frame(df, tsv)


def test_validate_frame_rejects_nonpositive_norm_reads(tmp_path):
    tsv = tmp_path / "bad2.tsv"
    _write_tsv(tsv, [_row(norm_reads=0)])
    df = sm.read_needed_columns(tsv)
    with pytest.raises(ValueError):
        sm.validate_frame(df, tsv)


def test_validate_frame_rejects_negative_pair_count(tmp_path):
    tsv = tmp_path / "bad3.tsv"
    _write_tsv(tsv, [_row(pair_count=-1)])
    df = sm.read_needed_columns(tsv)
    with pytest.raises(ValueError):
        sm.validate_frame(df, tsv)


def test_discover_tsv_files(tmp_path):
    (tmp_path / "a.tsv").write_text("x")
    (tmp_path / "b.tsv").write_text("x")
    (tmp_path / "c.txt").write_text("x")
    files = sm.discover_tsv_files(tmp_path)
    assert [f.name for f in files] == ["a.tsv", "b.tsv"]
