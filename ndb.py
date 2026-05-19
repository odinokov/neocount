"""Shared core: encoding, dense lookup, bloom helpers, mmap-backed NmerDB loader."""
import json
import math
import mmap
import struct
from collections import namedtuple

import numpy as np

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    _njit = None

MAGIC = b"NMERDB\x00\x01"
HEADER_STRUCT = struct.Struct('<8sBBBBQIIIQQIIII')
_Header = namedtuple('_Header', (
    'magic', 'ver_maj', 'ver_min', 'k', 'dtype',
    'n_entries', 'n_groups', 'cat_bytes', 'bloom_bytes',
    'kmer_off', 'group_off', 'flags',
    'source_crc32', 'build_timestamp', 'n_unique',
))

_ENCODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_DECODE = 'ACGT'
_AF_BREAKS = (0.001, 0.01, 0.1, 0.5)
_VC_NORM = {'Intergenic Region': 'IGR'}
AF_BIN_LABELS = ['<0.001', '0.001-0.01', '0.01-0.1', '0.1-0.5', '>=0.5', 'Unknown']

_BASE_TO_BITS = np.full(256, 255, dtype=np.uint8)
for _base, _bits in ((ord('A'), 0), (ord('C'), 1), (ord('G'), 2), (ord('T'), 3),
                     (ord('a'), 0), (ord('c'), 1), (ord('g'), 2), (ord('t'), 3)):
    _BASE_TO_BITS[_base] = _bits

__all__ = [
    'MAGIC', 'HEADER_STRUCT', 'AF_BIN_LABELS',
    'encode_kmer', 'decode_kmer', 'af_to_bin', 'normalize_variant_class',
    'NmerDB', 'sliding_kmers', 'lookup', 'lookup_grouped',
    'build_bloom_intmix', 'process_pair_batch_dense',
]


def encode_kmer(seq: str, k: int) -> int | None:
    """2-bit encode a DNA k-mer to uint32; returns None on length mismatch or non-ACGT character."""
    if len(seq) != k:
        return None
    val = 0
    for ch in seq.upper():
        enc = _ENCODE.get(ch)
        if enc is None:
            return None
        val = (val << 2) | enc
    return val


def decode_kmer(val: int, k: int) -> str:
    """Decode a 2-bit-encoded uint32 k-mer back to a DNA string."""
    chars = []
    for _ in range(k):
        chars.append(_DECODE[val & 3])
        val >>= 2
    return ''.join(reversed(chars))


def af_to_bin(af) -> int:
    """Map a germline AF value (str or float) to bin 0–4; returns 5 (Unknown) on NaN, out-of-range, or non-numeric."""
    try:
        f = float(af)
    except (ValueError, TypeError):
        return 5
    if math.isnan(f) or not 0.0 <= f <= 1.0:
        return 5
    for i, upper in enumerate(_AF_BREAKS):
        if f < upper:
            return i
    return 4


def normalize_variant_class(vc: str) -> str:
    """Normalise variant class labels (e.g. 'Intergenic Region' → 'IGR')."""
    return _VC_NORM.get(vc, vc)


_MIX_CONST_1 = np.uint64(0xbf58476d1ce4e5b9)
_MIX_CONST_2 = np.uint64(0x94d049bb133111eb)
_MIX_SEED_CONST = np.uint64(0x9e3779b97f4a7c15)


def _mix64_py(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return x & 0xFFFFFFFFFFFFFFFF


if _HAS_NUMBA:
    @_njit(cache=True)
    def _mix64_nb(x):
        x = np.uint64(x)
        x ^= x >> np.uint64(30)
        x *= _MIX_CONST_1
        x ^= x >> np.uint64(27)
        x *= _MIX_CONST_2
        x ^= x >> np.uint64(31)
        return x

    @_njit(cache=True)
    def _bloom_query_mask_intmix_nb(queries, data, n_bits, n_hashes, seed):
        out = np.empty(len(queries), dtype=np.bool_)
        n_bits_u = np.uint64(n_bits)
        seed_u = np.uint64(seed)
        for q_idx in range(len(queries)):
            val = np.uint64(queries[q_idx])
            h1 = _mix64_nb(val ^ seed_u)
            h2 = _mix64_nb((val + _MIX_SEED_CONST) ^ seed_u) | np.uint64(1)
            present = True
            for i in range(n_hashes):
                bit = (h1 + np.uint64(i) * h2) % n_bits_u
                byte = data[bit >> np.uint64(3)]
                if ((byte >> (bit & np.uint64(7))) & np.uint8(1)) == 0:
                    present = False
                    break
            out[q_idx] = present
        return out

    @_njit(cache=True)
    def _bloom_set_many_intmix_nb(kmers, data, n_bits, n_hashes, seed):
        n_bits_u = np.uint64(n_bits)
        seed_u = np.uint64(seed)
        for q_idx in range(len(kmers)):
            val = np.uint64(kmers[q_idx])
            h1 = _mix64_nb(val ^ seed_u)
            h2 = _mix64_nb((val + _MIX_SEED_CONST) ^ seed_u) | np.uint64(1)
            for i in range(n_hashes):
                bit = (h1 + np.uint64(i) * h2) % n_bits_u
                byte_idx = bit >> np.uint64(3)
                data[byte_idx] |= np.uint8(1 << (bit & np.uint64(7)))

    @_njit(cache=True)
    def _mark_groups_from_seq_nb(seqs, row, length, k, lookup_start, lookup_count,
                                 group_arr, seen_group, epoch):
        mask = np.uint32((1 << (2 * k)) - 1)
        cur = np.uint32(0)
        valid = 0
        found = False
        for pos in range(length):
            enc = _BASE_TO_BITS[seqs[row, pos]]
            if enc > 3:
                cur = np.uint32(0)
                valid = 0
                continue
            cur = ((cur << np.uint32(2)) | np.uint32(enc)) & mask
            valid += 1
            if valid < k:
                continue
            count = int(lookup_count[cur])
            if count == 0:
                continue
            start = int(lookup_start[cur])
            for off in range(count):
                seen_group[int(group_arr[start + off])] = epoch
            found = True
        return found

    @_njit(cache=True)
    def _count_common_groups_from_seq_nb(seqs, row, length, k, lookup_start,
                                         lookup_count, group_arr, seen_group,
                                         counted_group, group_counts, epoch):
        mask = np.uint32((1 << (2 * k)) - 1)
        cur = np.uint32(0)
        valid = 0
        found = False
        for pos in range(length):
            enc = _BASE_TO_BITS[seqs[row, pos]]
            if enc > 3:
                cur = np.uint32(0)
                valid = 0
                continue
            cur = ((cur << np.uint32(2)) | np.uint32(enc)) & mask
            valid += 1
            if valid < k:
                continue
            count = int(lookup_count[cur])
            if count == 0:
                continue
            start = int(lookup_start[cur])
            for off in range(count):
                gid = int(group_arr[start + off])
                if seen_group[gid] == epoch and counted_group[gid] != epoch:
                    group_counts[gid] += np.uint64(1)
                    counted_group[gid] = epoch
                    found = True
        return found

    @_njit(cache=True)
    def _observe_common_entries_from_seq_nb(seqs, row, length, k, lookup_start,
                                            lookup_count, group_arr, counted_group,
                                            observed_entries, epoch):
        mask = np.uint32((1 << (2 * k)) - 1)
        cur = np.uint32(0)
        valid = 0
        for pos in range(length):
            enc = _BASE_TO_BITS[seqs[row, pos]]
            if enc > 3:
                cur = np.uint32(0)
                valid = 0
                continue
            cur = ((cur << np.uint32(2)) | np.uint32(enc)) & mask
            valid += 1
            if valid < k:
                continue
            count = int(lookup_count[cur])
            if count == 0:
                continue
            start = int(lookup_start[cur])
            for off in range(count):
                entry_idx = start + off
                gid = int(group_arr[entry_idx])
                if counted_group[gid] == epoch:
                    observed_entries[entry_idx] = True

    @_njit(cache=True)
    def _count_pair_batch_dense_nb(seq1_batch, len1_batch, seq2_batch, len2_batch,
                                   n_pairs, k, lookup_start, lookup_count,
                                   group_arr, group_counts, observed_entries,
                                   seen_group, counted_group, epoch_start,
                                   emit_neomers):
        epoch = np.uint32(epoch_start)
        for pair_idx in range(n_pairs):
            seen_any = _mark_groups_from_seq_nb(
                seq1_batch, pair_idx, int(len1_batch[pair_idx]), k,
                lookup_start, lookup_count, group_arr, seen_group, epoch,
            )
            if seen_any:
                counted_any = _count_common_groups_from_seq_nb(
                    seq2_batch, pair_idx, int(len2_batch[pair_idx]), k,
                    lookup_start, lookup_count, group_arr, seen_group,
                    counted_group, group_counts, epoch,
                )
                if emit_neomers and counted_any:
                    _observe_common_entries_from_seq_nb(
                        seq1_batch, pair_idx, int(len1_batch[pair_idx]), k,
                        lookup_start, lookup_count, group_arr, counted_group,
                        observed_entries, epoch,
                    )
                    _observe_common_entries_from_seq_nb(
                        seq2_batch, pair_idx, int(len2_batch[pair_idx]), k,
                        lookup_start, lookup_count, group_arr, counted_group,
                        observed_entries, epoch,
                    )
            epoch += np.uint32(1)
        return epoch


def _bloom_query_mask_intmix_py(queries: np.ndarray, data: np.ndarray,
                                n_bits: int, n_hashes: int, seed: int) -> np.ndarray:
    out = np.empty(len(queries), dtype=bool)
    for q_idx, q in enumerate(queries):
        val = int(q)
        h1 = _mix64_py(val ^ seed)
        h2 = _mix64_py((val + 0x9e3779b97f4a7c15) ^ seed) | 1
        present = True
        for i in range(n_hashes):
            bit = (h1 + i * h2) % n_bits
            if not (int(data[bit >> 3]) >> (bit & 7)) & 1:
                present = False
                break
        out[q_idx] = present
    return out


def _bloom_set_many_intmix_py(kmers: np.ndarray, data: np.ndarray,
                              n_bits: int, n_hashes: int, seed: int) -> None:
    for q in kmers:
        val = int(q)
        h1 = _mix64_py(val ^ seed)
        h2 = _mix64_py((val + 0x9e3779b97f4a7c15) ^ seed) | 1
        for i in range(n_hashes):
            bit = (h1 + i * h2) % n_bits
            data[bit >> 3] |= np.uint8(1 << (bit & 7))


def build_bloom_intmix(kmers: np.ndarray, n_bits: int, n_hashes: int, seed: int) -> bytes:
    """Build an int-mixer bloom byte string for uint32 k-mers."""
    data = np.zeros((n_bits + 7) // 8, dtype=np.uint8)
    if len(kmers):
        kmers = np.asarray(kmers, dtype='<u4')
        if _HAS_NUMBA:
            _bloom_set_many_intmix_nb(kmers, data, n_bits, n_hashes, seed)
        else:
            _bloom_set_many_intmix_py(kmers, data, n_bits, n_hashes, seed)
    return data.tobytes()


class _Bloom:
    __slots__ = ('data', 'data_arr', 'n_bits', 'n_hashes', 'seed', 'hash_func')

    def __init__(self, data: bytearray, n_hashes: int, seed: int,
                 n_bits: int | None = None, hash_func: str = 'intmix64-v1'):
        if len(data) == 0:
            raise ValueError("bloom data must be non-empty")
        if hash_func != 'intmix64-v1':
            raise ValueError(f"unsupported bloom hash '{hash_func}'")
        if n_bits is None:
            raise ValueError("intmix64-v1 bloom requires exact m_bits metadata")
        self.data = data
        self.data_arr = np.frombuffer(data, dtype=np.uint8)
        self.n_bits = n_bits
        self.n_hashes = n_hashes
        self.seed = seed
        self.hash_func = hash_func

    def filter(self, queries: np.ndarray) -> np.ndarray:
        """Return a boolean mask for queries that may be present."""
        if len(queries) == 0:
            return np.empty(0, dtype=bool)
        if _HAS_NUMBA:
            return _bloom_query_mask_intmix_nb(
                np.asarray(queries, dtype='<u4'), self.data_arr,
                self.n_bits, self.n_hashes, self.seed,
            )
        return _bloom_query_mask_intmix_py(
            np.asarray(queries, dtype='<u4'), self.data_arr,
            self.n_bits, self.n_hashes, self.seed,
        )


def _check_ndb_bounds(hdr, fsize: int):
    """Raise SystemExit if any array offset lies outside the file; all four checks in one place."""
    kmer_end = hdr.kmer_off + hdr.n_entries * 4
    if kmer_end > fsize:
        raise SystemExit(f"ndb corrupt: kmer_array_offset {hdr.kmer_off} exceeds file size {fsize}")
    if hdr.group_off + hdr.n_entries * 2 > fsize:
        raise SystemExit(f"ndb corrupt: group_array_offset {hdr.group_off} exceeds file size {fsize}")
    if hdr.group_off < kmer_end:
        raise SystemExit(f"ndb corrupt: group_array_offset {hdr.group_off} overlaps kmer array (end={kmer_end})")
    if 64 + hdr.cat_bytes + hdr.bloom_bytes > hdr.kmer_off:
        raise SystemExit(
            f"ndb corrupt: catalog+bloom region ({64 + hdr.cat_bytes + hdr.bloom_bytes})"
            f" overruns kmer_array_offset {hdr.kmer_off}"
        )


def _parse_header(mm, path: str):
    """Parse and validate .ndb header from mmap; raises SystemExit on any format violation."""
    try:
        hdr = _Header._make(HEADER_STRUCT.unpack_from(mm, 0))
    except struct.error as e:
        raise SystemExit(f"'{path}' is too small to be a valid .ndb file: {e}")
    if hdr.magic != MAGIC:
        raise SystemExit(f"Not a valid .ndb file: {path}")
    if hdr.ver_maj != 1:
        raise SystemExit(f"Unsupported .ndb version {hdr.ver_maj}: {path}")
    if hdr.dtype != 0:
        raise SystemExit("k=17 uint64 dtype not supported — rebuild with k≤16")
    if hdr.k not in range(11, 17):
        raise SystemExit(f"Unsupported k={hdr.k} (supported: 11–16)")
    if hdr.flags & 4:
        raise SystemExit("RC-inclusive .ndb not supported in this version")
    _check_ndb_bounds(hdr, len(mm))
    return hdr


def _load_catalog(mm, hdr):
    """Parse and validate catalog JSON from mmap; raises SystemExit on format error."""
    try:
        cat_json = bytes(mm[64:64 + hdr.cat_bytes]).decode()
    except UnicodeDecodeError as e:
        raise SystemExit(f"ndb corrupt: catalog is not valid UTF-8: {e}")
    try:
        cat = json.loads(cat_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ndb corrupt: unparseable catalog JSON: {e}")
    n_catalog_groups = len(cat.get('groups', []))
    if n_catalog_groups != hdr.n_groups:
        raise SystemExit(
            f"ndb corrupt: header declares {hdr.n_groups} groups but catalog has {n_catalog_groups}"
        )
    return cat


def _load_bloom_filter(mm, hdr, cat):
    """Build _Bloom from mmap if bloom present, else None; raises SystemExit on unsupported format."""
    if not hdr.bloom_bytes:
        return None
    bp = cat.get('bloom_params', {})
    hash_func = bp.get('hash_func')
    if hash_func != 'intmix64-v1':
        raise SystemExit(
            f"Unsupported legacy bloom hash '{hash_func or 'unknown'}'; "
            "rebuild this .ndb with the current builder"
        )
    bloom_start = 64 + hdr.cat_bytes
    return _Bloom(
        bytearray(mm[bloom_start:bloom_start + hdr.bloom_bytes]),
        bp.get('n_hashes', 7),
        bp.get('seed', 42),
        bp.get('m_bits'),
        hash_func,
    )


def _build_dense_lookup(kmer_arr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build direct-address lookup tables mapping encoded k-mer -> entry slice."""
    if k > 15:
        raise SystemExit("dense lookup currently supports k<=15 on 32 GB hosts")
    space_size = 1 << (2 * k)
    lookup_start = np.zeros(space_size, dtype=np.uint32)
    lookup_count = np.zeros(space_size, dtype=np.uint16)
    if len(kmer_arr):
        unique_kmers, first_idx, counts = np.unique(kmer_arr, return_index=True, return_counts=True)
        max_count = int(counts.max()) if len(counts) else 0
        if max_count > np.iinfo(np.uint16).max:
            raise SystemExit(f"dense lookup cannot store {max_count} groups for one k-mer")
        lookup_start[unique_kmers] = first_idx.astype(np.uint32)
        lookup_count[unique_kmers] = counts.astype(np.uint16)
    return lookup_start, lookup_count, space_size


class NmerDB:
    __slots__ = (
        'k', 'kmer_arr', 'group_arr', 'bloom', 'catalog',
        'flags', 'source_crc32', 'build_timestamp', 'n_unique',
        'lookup_start', 'lookup_count', 'lookup_space_size', 'lookup_bytes',
        '_mm', '_fh',
    )

    def __init__(self, path: str, preload: bool = True, dense: bool = True):
        try:
            self._fh = open(path, 'rb')
        except (FileNotFoundError, PermissionError, OSError) as e:
            raise SystemExit(f"Cannot open .ndb file: {e}")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError) as e:
            self._fh.close()
            raise SystemExit(f"Cannot mmap '{path}': {e}")
        try:
            hdr = _parse_header(self._mm, path)
            cat = _load_catalog(self._mm, hdr)
            self.bloom = _load_bloom_filter(self._mm, hdr, cat)
            self.k = hdr.k
            self.flags = hdr.flags
            self.source_crc32 = hdr.source_crc32
            self.build_timestamp = hdr.build_timestamp
            self.n_unique = hdr.n_unique
            self.catalog = {g['id']: g for g in cat['groups']}
            self.kmer_arr = np.frombuffer(self._mm, dtype='<u4', count=hdr.n_entries, offset=hdr.kmer_off)
            self.group_arr = np.frombuffer(self._mm, dtype='<u2', count=hdr.n_entries, offset=hdr.group_off)
            if preload:
                self.kmer_arr = self.kmer_arr.copy()
                self.group_arr = self.group_arr.copy()
            if dense:
                self.lookup_start, self.lookup_count, self.lookup_space_size = _build_dense_lookup(self.kmer_arr, self.k)
                self.lookup_bytes = self.lookup_start.nbytes + self.lookup_count.nbytes
            else:
                self.lookup_start = None
                self.lookup_count = None
                self.lookup_space_size = 0
                self.lookup_bytes = 0
        except SystemExit:
            self._mm.close()
            self._fh.close()
            raise

    def close(self):
        self.kmer_arr = None   # release mmap buffer reference before closing
        self.group_arr = None
        self.lookup_start = None
        self.lookup_count = None
        self._mm.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def sliding_kmers(seq: str | None, k: int) -> np.ndarray:
    """Extract all k-mers from seq as uint32 array in read order; resets on non-ACGT.

    Lookup deduplicates these per-read query arrays before bloom/search.
    """
    if seq is None:
        return np.empty(0, dtype='<u4')
    mask = (1 << (2 * k)) - 1
    cur = valid = 0
    out = []
    for ch in seq.upper():
        enc = _ENCODE.get(ch)
        if enc is None:
            cur = valid = 0
            continue
        cur = ((cur << 2) | enc) & mask
        valid += 1
        if valid >= k:
            out.append(cur)
    return np.array(out, dtype='<u4')


def _scan_hits(queries: np.ndarray, db: NmerDB):
    """Yield (kmer_val, group_id) for every query k-mer found in db.

    Shared core for lookup() and lookup_grouped(). Handles bloom pre-filter,
    np.searchsorted, and consecutive-duplicate walk for cross-cancer neomers.
    """
    if len(queries) == 0 or len(db.kmer_arr) == 0:
        return
    queries = np.unique(queries)
    if db.bloom:
        queries = queries[db.bloom.filter(queries)]
    if len(queries) == 0:
        return
    n = len(db.kmer_arr)
    idxs = np.searchsorted(db.kmer_arr, queries, side='left')
    # Clamp out-of-range indices to 0 for safe indexing; equality check filters them out.
    safe = np.where(idxs < n, idxs, 0)
    valid = (idxs < n) & (db.kmer_arr[safe] == queries)
    for i in safe[valid]:
        kmer = int(db.kmer_arr[i])
        j = int(i)
        while j < n and db.kmer_arr[j] == kmer:
            yield kmer, int(db.group_arr[j])
            j += 1


def lookup_grouped(queries: np.ndarray, db: NmerDB) -> dict:
    """Return {group_id: set[kmer_val]} for k-mers found in db."""
    result: dict = {}
    for kmer, gid in _scan_hits(queries, db):
        result.setdefault(gid, set()).add(kmer)
    return result


def lookup(queries: np.ndarray, db: NmerDB) -> set:
    """Return set of group IDs (int) for k-mers found in db (AND logic: caller intersects two sets)."""
    return {gid for _kmer, gid in _scan_hits(queries, db)}


def process_pair_batch_dense(seq1_batch: np.ndarray, len1_batch: np.ndarray,
                             seq2_batch: np.ndarray, len2_batch: np.ndarray,
                             n_pairs: int, db: NmerDB, group_counts: np.ndarray,
                             observed_entries: np.ndarray, seen_group: np.ndarray,
                             counted_group: np.ndarray, epoch_start: int,
                             emit_neomers: bool) -> int:
    """Count a batch of read pairs using direct-address dense lookup tables."""
    if not _HAS_NUMBA:
        raise SystemExit("Numba is required for dense batched counting")
    if db.lookup_start is None or db.lookup_count is None:
        raise SystemExit("dense lookup tables are not loaded")
    return int(_count_pair_batch_dense_nb(
        seq1_batch, len1_batch, seq2_batch, len2_batch, n_pairs,
        db.k, db.lookup_start, db.lookup_count, db.group_arr,
        group_counts, observed_entries, seen_group, counted_group,
        epoch_start, emit_neomers,
    ))
