"""Regression tests for build_neomersdb.py: parsing, remap/sort, and build->load."""
import io
import tarfile
import types

import numpy as np
import pytest

import build_neomersdb as bld
import ndb


# ---- small pure helpers ----------------------------------------------------

def test_infer_k():
    assert bld._infer_k(["ACGTACGTACG"]) == 11


@pytest.mark.parametrize("name,expected_suffix", [
    ("neomers_11.csv.tar.gz", "neomers_11.ndb"),
    ("foo.csv.gz", "foo.ndb"),
    ("foo.csv", "foo.ndb"),
    ("foo.dat", "foo.ndb"),
])
def test_default_ndb_path(name, expected_suffix):
    assert bld._default_ndb_path(name).endswith(expected_suffix)


def test_fmt_mb():
    assert bld._fmt_mb(1_000_000) == "1.0 MB"
    assert bld._fmt_mb(150_000_000) == "150 MB"


def test_crc32_reader_accumulates():
    raw = io.BytesIO(b"hello world")
    r = bld._CRC32Reader(raw)
    data = r.read()
    assert data == b"hello world"
    import zlib
    assert r.crc32 == zlib.crc32(b"hello world") & 0xFFFFFFFF


# ---- _process_rows ---------------------------------------------------------

def _row(kmer, ct, org, vc, af):
    # 14 columns; only indices 0,1,2,4,5 are read
    row = [""] * 14
    row[0], row[1], row[2], row[4], row[5] = kmer, ct, org, vc, af
    return row


def test_process_rows_groups_and_counts():
    rows = [
        _row("ACGTACGTACG", "LUAD", "Lung", "Missense", "0.0005"),
        _row("ACGTACGTACG", "LUAD", "Lung", "Missense", "0.0005"),  # same group
        _row("TTTTTTTTTTT", "BRCA", "Breast", "Nonsense", "0.2"),
        _row("ACG", "LUAD", "Lung", "Missense", "0.0005"),          # wrong k -> skip
        _row("ACGTACGTACN", "LUAD", "Lung", "Missense", "0.0005"),  # bad base -> skip
        ["only", "three", "cols"],                                  # wrong col count
    ]
    arr, group_dict, stats = bld._process_rows(iter(rows), 11)
    assert len(arr) == 3  # three valid k-mers
    assert len(group_dict) == 2
    assert stats["n_skipped_k"] == 1
    assert stats["n_skipped_base"] == 1
    assert stats["n_skipped_col"] == 1


# ---- _remap_and_sort -------------------------------------------------------

def test_remap_and_sort_dedups_and_sorts():
    arr = np.array([(9, 1), (5, 0), (5, 0)], dtype=[("k", "<u4"), ("g", "<u2")])
    group_dict = {("a",): 0, ("b",): 1}
    out, canon, n_dup = bld._remap_and_sort(arr, group_dict)
    assert n_dup == 1  # the duplicate (5,0)
    assert list(out["k"]) == [5, 9]
    assert len(canon) == 2


# ---- build -> load roundtrip ----------------------------------------------

def _make_csv_targz(tmp_path):
    header = ",".join([f"c{i}" for i in range(14)])
    rows = [
        _row("ACGTACGTACG", "LUAD", "Lung", "Missense", "0.0005"),
        _row("TTTTTTTTTTT", "BRCA", "Breast", "Nonsense", "0.2"),
        _row("GATTACAGATT", "LUAD", "Lung", "Missense", "0.0005"),
    ]
    csv_text = header + "\n" + "\n".join(",".join(r) for r in rows) + "\n"
    csv_bytes = csv_text.encode()

    targz = tmp_path / "neomers_11.csv.tar.gz"
    with tarfile.open(targz, "w:gz") as tf:
        info = tarfile.TarInfo("neomers_11.csv")
        info.size = len(csv_bytes)
        tf.addfile(info, io.BytesIO(csv_bytes))
    return targz


def test_build_then_load_roundtrip(tmp_path):
    targz = _make_csv_targz(tmp_path)
    out = tmp_path / "out.ndb"
    args = types.SimpleNamespace(input=str(targz), output=str(out))
    bld._run_pipeline(args)
    assert out.exists()

    with ndb.NmerDB(str(out)) as db:
        assert db.k == 11
        assert len(db.kmer_arr) == 3
        # two distinct (cancer_type, organ, variant_class, af_bin) groups
        assert len(db.catalog) == 2
        # a known k-mer should resolve to a group
        q = ndb.sliding_kmers("ACGTACGTACG", 11)
        assert len(ndb.lookup(q, db)) >= 1
