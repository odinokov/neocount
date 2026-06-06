"""Regression tests for ndb.py pure logic: encoding, AF binning, bloom, lookup."""
import math
import types

import numpy as np
import pytest

import ndb


# ---- encode/decode ---------------------------------------------------------

def test_encode_kmer_basic():
    # A=0 C=1 G=2 T=3 -> 0b00_01_10_11 = 27
    assert ndb.encode_kmer("ACGT", 4) == 27


def test_encode_kmer_lowercase_uppercased():
    assert ndb.encode_kmer("acgt", 4) == 27


def test_encode_kmer_wrong_length_returns_none():
    assert ndb.encode_kmer("ACG", 4) is None
    assert ndb.encode_kmer("ACGTA", 4) is None


def test_encode_kmer_non_acgt_returns_none():
    assert ndb.encode_kmer("ACGN", 4) is None


@pytest.mark.parametrize("seq", ["A" * 11, "ACGTACGTACG", "TTTTTTTTTTT", "GATTACAGATT"])
def test_encode_decode_roundtrip(seq):
    k = len(seq)
    assert ndb.decode_kmer(ndb.encode_kmer(seq, k), k) == seq


# ---- af_to_bin -------------------------------------------------------------

@pytest.mark.parametrize("af,expected", [
    ("0.0005", 0),
    ("0.001", 1),    # not < 0.001, < 0.01
    ("0.005", 1),
    ("0.01", 2),
    ("0.05", 2),
    ("0.1", 3),
    ("0.3", 3),
    ("0.5", 4),      # not < 0.5
    ("0.9", 4),
    ("1.0", 4),
    (0.0, 0),
    ("nan", 5),
    ("-0.1", 5),
    ("2.0", 5),
    ("abc", 5),
    (None, 5),
])
def test_af_to_bin(af, expected):
    assert ndb.af_to_bin(af) == expected


# ---- normalize_variant_class ----------------------------------------------

def test_normalize_variant_class():
    assert ndb.normalize_variant_class("Intergenic Region") == "IGR"
    assert ndb.normalize_variant_class("Missense") == "Missense"


def test_af_bin_meta_matches_canonical_literal():
    # Guards the single-source derivation: must equal the historical catalog literal.
    assert ndb.AF_BIN_META == [
        {'id': 0, 'label': '<0.001',     'max_af': 0.001},
        {'id': 1, 'label': '0.001-0.01', 'max_af': 0.01},
        {'id': 2, 'label': '0.01-0.1',   'max_af': 0.1},
        {'id': 3, 'label': '0.1-0.5',    'max_af': 0.5},
        {'id': 4, 'label': '>=0.5',      'max_af': 1.0},
        {'id': 5, 'label': 'Unknown',    'max_af': None},
    ]


# ---- _mix64 determinism + numba/py parity ----------------------------------

def test_mix64_deterministic():
    assert ndb._mix64_py(12345) == ndb._mix64_py(12345)
    assert ndb._mix64_py(0) == ndb._mix64_py(0)


# ---- sliding_kmers ---------------------------------------------------------

def test_sliding_kmers_count_and_value():
    out = ndb.sliding_kmers("ACGTACGT", 4)
    assert len(out) == 5
    assert out[0] == 27  # ACGT


def test_sliding_kmers_resets_on_non_acgt():
    # N resets the rolling window
    out = ndb.sliding_kmers("ACGTNACGT", 4)
    assert len(out) == 2  # one before N is impossible at pos<4; two windows total


def test_sliding_kmers_none_returns_empty():
    assert len(ndb.sliding_kmers(None, 4)) == 0


# ---- bloom roundtrip (no false negatives) ----------------------------------

def test_bloom_no_false_negatives():
    kmers = np.array([27, 100, 255, 4096, 65535], dtype="<u4")
    n_bits, n_hashes, seed = 1024, 7, 42
    data = ndb.build_bloom_intmix(kmers, n_bits, n_hashes, seed)
    bloom = ndb._Bloom(bytearray(data), n_hashes, seed, n_bits=n_bits)
    mask = bloom.filter(kmers)
    assert mask.all()  # every inserted k-mer must be reported present


def test_bloom_empty_query():
    data = ndb.build_bloom_intmix(np.array([1, 2, 3], dtype="<u4"), 256, 7, 42)
    bloom = ndb._Bloom(bytearray(data), 7, 42, n_bits=256)
    assert len(bloom.filter(np.empty(0, dtype="<u4"))) == 0


def test_bloom_py_path_no_false_negatives():
    kmers = np.array([27, 100, 255, 4096, 65535], dtype="<u4")
    n_bits, n_hashes, seed = 1024, 7, 42
    data = np.zeros((n_bits + 7) // 8, dtype=np.uint8)
    ndb._bloom_set_many_intmix_py(kmers, data, n_bits, n_hashes, seed)
    mask = ndb._bloom_query_mask_intmix_py(kmers, data, n_bits, n_hashes, seed)
    assert mask.all()


def test_bloom_py_numba_parity():
    if not ndb._HAS_NUMBA:
        pytest.skip("numba absent")
    kmers = np.array([27, 100, 255, 4096, 65535], dtype="<u4")
    queries = np.array([27, 999, 4096, 123456], dtype="<u4")
    n_bits, n_hashes, seed = 1024, 7, 42
    # same set of bits must be produced by both implementations
    data_py = np.zeros((n_bits + 7) // 8, dtype=np.uint8)
    ndb._bloom_set_many_intmix_py(kmers, data_py, n_bits, n_hashes, seed)
    data_nb = np.frombuffer(ndb.build_bloom_intmix(kmers, n_bits, n_hashes, seed), dtype=np.uint8)
    assert data_py.tobytes() == data_nb.tobytes()
    # same query verdicts from both query implementations
    m_py = ndb._bloom_query_mask_intmix_py(queries, data_py, n_bits, n_hashes, seed)
    m_nb = ndb._bloom_query_mask_intmix_nb(queries, data_nb, n_bits, n_hashes, seed)
    assert np.array_equal(m_py, m_nb)


# ---- dense lookup ----------------------------------------------------------

def test_build_dense_lookup():
    kmer_arr = np.array([5, 5, 9], dtype="<u4")  # sorted, k=4 -> space 256
    start, count, space = ndb._build_dense_lookup(kmer_arr, 4)
    assert space == 1 << 8
    assert count[5] == 2
    assert count[9] == 1
    assert start[5] == 0
    assert start[9] == 2


# ---- lookup / lookup_grouped via a lightweight db stand-in ------------------

def _fake_db():
    db = types.SimpleNamespace()
    # k-mers sorted ascending; two groups share kmer 5
    db.kmer_arr = np.array([5, 5, 9], dtype="<u4")
    db.group_arr = np.array([0, 1, 1], dtype="<u2")
    db.bloom = None
    return db


def test_lookup_returns_group_ids():
    db = _fake_db()
    q = np.array([5], dtype="<u4")
    assert ndb.lookup(q, db) == {0, 1}


def test_lookup_grouped():
    db = _fake_db()
    q = np.array([5, 9], dtype="<u4")
    grouped = ndb.lookup_grouped(q, db)
    assert grouped == {0: {5}, 1: {5, 9}}


def test_lookup_miss_returns_empty():
    db = _fake_db()
    assert ndb.lookup(np.array([1234], dtype="<u4"), db) == set()
