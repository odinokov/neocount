"""Regression tests for count_neomers.py: filters, batching helpers, counting."""
import io
import tarfile
import types
from collections import OrderedDict

import numpy as np
import pytest

import count_neomers as cn
import ndb


# ---- region parsing --------------------------------------------------------

def test_parse_norm_regions():
    d = cn._parse_norm_regions("chr2:95000000-242000000,chr19:0-24200000")
    assert d["chr2"] == (95000000, 242000000)
    assert d["chr19"] == (0, 24200000)


def test_build_autosome_set():
    s = cn._build_autosome_set()
    assert "1" in s and "chr1" in s and "22" in s and "chr22" in s
    assert "chrX" not in s and "23" not in s


def _read(reference_name="chr2", reference_start=100_000_000, flag=0, mapping_quality=30):
    return types.SimpleNamespace(
        reference_name=reference_name,
        reference_start=reference_start,
        flag=flag,
        mapping_quality=mapping_quality,
    )


def test_in_norm_region_default():
    norm = cn._parse_norm_regions(cn._DEFAULT_NORM_REGIONS)
    auto = cn._build_autosome_set()
    assert cn._in_norm_region(_read("chr2", 100_000_000), norm, auto, False) is True
    assert cn._in_norm_region(_read("chr2", 1_000), norm, auto, False) is False  # before region
    assert cn._in_norm_region(_read("chrX", 100), norm, auto, False) is False


def test_in_norm_region_autosomes():
    norm = cn._parse_norm_regions(cn._DEFAULT_NORM_REGIONS)
    auto = cn._build_autosome_set()
    assert cn._in_norm_region(_read("chr7", 5), norm, auto, True) is True
    assert cn._in_norm_region(_read("chrX", 5), norm, auto, True) is False


def test_in_norm_region_no_reference_name():
    norm = cn._parse_norm_regions(cn._DEFAULT_NORM_REGIONS)
    auto = cn._build_autosome_set()
    assert cn._in_norm_region(_read(None, 5), norm, auto, False) is False


# ---- read filters ----------------------------------------------------------

def test_is_usable_read():
    assert cn._is_usable_read(_read(flag=0, mapping_quality=30), 5) is True
    assert cn._is_usable_read(_read(flag=4, mapping_quality=30), 5) is False    # unmapped
    assert cn._is_usable_read(_read(flag=1024, mapping_quality=30), 5) is False  # dup
    assert cn._is_usable_read(_read(flag=0, mapping_quality=3), 5) is False      # low mapq


@pytest.mark.parametrize("tlen,read_len,k,expected", [
    (0, 100, 11, True),       # tlen 0 passes through
    (150, 100, 11, True),     # 150 <= 200-11
    (190, 100, 11, False),    # 190 > 189
])
def test_tlen_ok(tlen, read_len, k, expected):
    assert cn._tlen_ok(tlen, read_len, k) is expected


# ---- pending cache ---------------------------------------------------------

def test_enqueue_read_eviction():
    pending = OrderedDict()
    assert cn._enqueue_read(pending, "a", "AAA", 2) == 0
    assert cn._enqueue_read(pending, "b", "BBB", 2) == 0
    assert cn._enqueue_read(pending, "c", "CCC", 2) == 1  # evicts "a"
    assert "a" not in pending
    assert list(pending.keys()) == ["b", "c"]


# ---- batch helpers ---------------------------------------------------------

def test_next_batch_width():
    assert cn._next_batch_width(256, 300) == 512
    assert cn._next_batch_width(256, 256) == 256
    assert cn._next_batch_width(256, 1100) == 2048


def test_store_seq():
    dst = np.zeros(16, dtype=np.uint8)
    n = cn._store_seq(dst, "ACGT")
    assert n == 4
    assert list(dst[:4]) == [ord("A"), ord("C"), ord("G"), ord("T")]


def test_alloc_pair_batch_shapes():
    s1, s2, l1, l2 = cn._alloc_pair_batch(8, 32)
    assert s1.shape == (8, 32) and s2.shape == (8, 32)
    assert l1.shape == (8,) and l2.shape == (8,)


# ---- _collect_observed_neomers --------------------------------------------

def test_collect_observed_neomers():
    db = types.SimpleNamespace(
        kmer_arr=np.array([5, 9], dtype="<u4"),
        group_arr=np.array([0, 1], dtype="<u2"),
    )
    observed = np.array([True, False])
    assert cn._collect_observed_neomers(db, observed) == {0: {5}}
    assert cn._collect_observed_neomers(db, None) == {}


# ---- counting integration via a real tiny .ndb -----------------------------

def _build_db(tmp_path):
    header = ",".join([f"c{i}" for i in range(14)])
    def row(kmer, ct, org, vc, af):
        r = [""] * 14
        r[0], r[1], r[2], r[4], r[5] = kmer, ct, org, vc, af
        return ",".join(r)
    rows = [
        row("ACGTACGTACG", "LUAD", "Lung", "Missense", "0.0005"),
        row("TTTTTTTTTTT", "BRCA", "Breast", "Nonsense", "0.2"),
    ]
    csv_bytes = (header + "\n" + "\n".join(rows) + "\n").encode()
    targz = tmp_path / "neomers_11.csv.tar.gz"
    with tarfile.open(targz, "w:gz") as tf:
        info = tarfile.TarInfo("neomers_11.csv")
        info.size = len(csv_bytes)
        tf.addfile(info, io.BytesIO(csv_bytes))
    import build_neomersdb as bld
    out = tmp_path / "out.ndb"
    bld._run_pipeline(types.SimpleNamespace(input=str(targz), output=str(out)))
    return ndb.NmerDB(str(out))


def test_count_pair_batch_dense_counts_shared_group(tmp_path):
    db = _build_db(tmp_path)
    try:
        n_groups = max(db.catalog) + 1
        # both mates contain the group-0 k-mer ACGTACGTACG
        seq = "ACGTACGTACG"
        width = 16
        seq1 = np.zeros((1, width), dtype=np.uint8)
        seq2 = np.zeros((1, width), dtype=np.uint8)
        data = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        seq1[0, : len(data)] = data
        seq2[0, : len(data)] = data
        len1 = np.array([len(data)], dtype=np.uint16)
        len2 = np.array([len(data)], dtype=np.uint16)
        group_counts = np.zeros(n_groups, dtype=np.uint64)
        observed = np.zeros(len(db.kmer_arr), dtype=np.bool_)
        seen = np.zeros(n_groups, dtype=np.uint32)
        counted = np.zeros(n_groups, dtype=np.uint32)
        cn.process_pair_batch_dense(
            seq1, len1, seq2, len2, 1, db,
            group_counts, observed, seen, counted, 1, True,
        )
        assert int(group_counts.sum()) == 1
        assert observed.any()
    finally:
        db.close()
