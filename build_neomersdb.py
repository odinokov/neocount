#!/usr/bin/env python3
"""Build a .ndb binary index from a neomerDB CSV.tar.gz archive."""

import argparse
import csv
import io
import itertools
import json
import platform
import sys
import tarfile
import time
import zlib
from importlib import metadata
from pathlib import Path

import numpy as np

from ndb import (MAGIC, HEADER_STRUCT, encode_kmer, af_to_bin,
                 normalize_variant_class)

_VERSION = 'v0.1'
_CHUNK = 5_000_000   # rows per numpy chunk during streaming


class _CRC32Reader:
    """Binary wrapper that accumulates CRC32 of all bytes read through it."""
    def __init__(self, fh):
        self._fh = fh
        self.crc32 = 0

    def read(self, n=-1):
        data = self._fh.read(n) if n >= 0 else self._fh.read()
        if data:
            self.crc32 = zlib.crc32(data, self.crc32) & 0xFFFFFFFF
        return data

    def readline(self):
        data = self._fh.readline()
        if data:
            self.crc32 = zlib.crc32(data, self.crc32) & 0xFFFFFFFF
        return data

    def readable(self):
        return self._fh.readable()

    def writable(self):
        return self._fh.writable() if hasattr(self._fh, 'writable') else False

    def seekable(self):
        return self._fh.seekable() if hasattr(self._fh, 'seekable') else False

    @property
    def closed(self):
        return self._fh.closed

    def close(self):
        return self._fh.close()

    def flush(self):
        if hasattr(self._fh, 'flush'):
            self._fh.flush()


def _open_csv(tar_path: str):
    """Return (tf, crc_reader, csv_reader) for the single .csv member in the archive.

    Caller is responsible for closing tf.  CRC32 accumulates as rows are read.
    """
    tf = tarfile.open(tar_path, 'r:gz')
    members = [m for m in tf.getmembers() if m.name.endswith('.csv')]
    if len(members) == 0:
        tf.close()
        raise SystemExit(f"No .csv member found in {tar_path}")
    if len(members) > 1:
        names = [m.name for m in members]
        tf.close()
        raise SystemExit(f"Multiple .csv members in {tar_path}: {names}")
    raw_fh = tf.extractfile(members[0])
    if raw_fh is None:
        tf.close()
        raise SystemExit(f"Cannot extract .csv member from {tar_path} (symlink not supported)")
    crc_reader = _CRC32Reader(raw_fh)
    text_wrapper = io.TextIOWrapper(crc_reader, encoding='utf-8')
    reader = csv.reader(text_wrapper)
    return tf, crc_reader, reader


def _infer_k(first_data_row: list) -> int:
    return len(first_data_row[0].strip())


def _process_rows(reader, k: int, first_row=None):
    """Stream CSV rows into a numpy structured array.

    Returns (arr, group_dict, stats) where:
      arr       dtype=[('k','<u4'),('g','<u2')], unsorted, with temp group IDs
      group_dict  {(ct, org, vc, af_bin): temp_gid}
      stats       dict with skip counts
    """
    group_dict = {}
    n_rows = n_skipped_col = n_skipped_k = n_skipped_base = n_valid = 0

    buf = np.empty(_CHUNK, dtype=[('k', '<u4'), ('g', '<u2')])
    n_buf = 0
    chunks = []

    source = itertools.chain([first_row], reader) if first_row is not None else reader
    _log_step = 100_000 if sys.stderr.isatty() else 1_000_000
    for row in source:
        n_rows += 1
        if len(row) != 14:
            n_skipped_col += 1
            continue
        kmer_str = row[0].strip().upper()
        if len(kmer_str) != k:
            n_skipped_k += 1
            continue
        kmer_val = encode_kmer(kmer_str, k)
        if kmer_val is None:
            n_skipped_base += 1
            continue

        ct  = row[1].strip()
        org = row[2].strip()
        vc  = normalize_variant_class(row[4].strip())
        af_bin = af_to_bin(row[5].strip())

        key = (ct, org, vc, af_bin)
        if key not in group_dict:
            if len(group_dict) >= 65535:
                raise SystemExit("Too many groups (>65535); uint16 group IDs support at most 65535")
            group_dict[key] = len(group_dict)  # monotonic temp id = insertion count

        buf[n_buf] = (kmer_val, group_dict[key])
        n_buf += 1
        n_valid += 1
        if n_valid % _log_step == 0:
            print(f"\r  {n_valid:,} k-mers read...", end='', file=sys.stderr, flush=True)
        if n_buf == _CHUNK:
            chunks.append(buf.copy())
            n_buf = 0

    if n_buf:
        chunks.append(buf[:n_buf].copy())

    if n_valid > 0:
        print(f"\r  {n_valid:,} k-mers read", file=sys.stderr)

    arr = np.concatenate(chunks) if chunks else np.empty(0, dtype=[('k', '<u4'), ('g', '<u2')])
    chunks.clear()
    stats = dict(n_rows=n_rows, n_skipped_col=n_skipped_col,
                 n_skipped_k=n_skipped_k, n_skipped_base=n_skipped_base)
    return arr, group_dict, stats


def _remap_and_sort(arr, group_dict: dict):
    """Remap temp group IDs → canonical sorted IDs; sort; dedup consecutive pairs.

    Returns (arr, canonical_groups, n_duplicates_collapsed).
    canonical_groups: {canonical_gid: (ct, org, vc, af_bin)}
    """
    if len(group_dict) > 65535:
        raise SystemExit(f"Too many groups ({len(group_dict)}); uint16 group IDs support at most 65535")
    sorted_keys = sorted(group_dict)
    remap = np.array([group_dict[k] for k in sorted_keys], dtype=np.uint16)
    inv = np.empty(len(remap), dtype=np.uint16)
    inv[remap] = np.arange(len(remap), dtype=np.uint16)
    arr['g'][:] = inv[arr['g']]

    arr.sort(order=('k', 'g'))

    n_dup = 0
    if len(arr) > 1:
        mask = np.ones(len(arr), dtype=bool)
        mask[1:] = (arr['k'][1:] != arr['k'][:-1]) | (arr['g'][1:] != arr['g'][:-1])
        n_dup = int((~mask).sum())
        arr = arr[mask]

    canonical_groups = {i: key for i, key in enumerate(sorted_keys)}
    return arr, canonical_groups, n_dup


def _compute_group_stats(kmer_arr, group_arr, n_groups: int, unique_kmers, kmer_counts):
    """Compute n_kmers and n_exclusive_kmers per group after dedup."""
    n_kmers = np.bincount(group_arr.astype(np.int64), minlength=n_groups)

    excl_vals = unique_kmers[kmer_counts == 1]
    if len(excl_vals):
        excl_idxs = np.searchsorted(kmer_arr, excl_vals)
        excl_gids = group_arr[excl_idxs]
        n_exclusive = np.bincount(excl_gids.astype(np.int64), minlength=n_groups)
    else:
        n_exclusive = np.zeros(n_groups, dtype=np.int64)

    return n_kmers, n_exclusive


_AF_BIN_META = [
    {'id': 0, 'label': '<0.001',     'max_af': 0.001},
    {'id': 1, 'label': '0.001-0.01', 'max_af': 0.01},
    {'id': 2, 'label': '0.01-0.1',   'max_af': 0.1},
    {'id': 3, 'label': '0.1-0.5',    'max_af': 0.5},
    {'id': 4, 'label': '>=0.5',      'max_af': 1.0},
    {'id': 5, 'label': 'Unknown',     'max_af': None},
]


def _build_catalog_json(in_path: str, canonical_groups: dict, build_ts: int,
                        n_rows_raw: int, process_stats: dict, build_stats: dict) -> bytes:
    """Serialise group catalog + build metadata to JSON bytes."""
    n_kmers_arr = build_stats['n_kmers']
    n_excl_arr = build_stats['n_exclusive']

    group_list = [
        {
            'id': gid,
            'cancer_type': ct,
            'organ': org,
            'variant_class': vc,
            'af_bin': af_bin,
            'n_kmers': int(n_kmers_arr[gid]),
            'n_exclusive_kmers': int(n_excl_arr[gid]),
        }
        for gid, (ct, org, vc, af_bin) in sorted(canonical_groups.items())
    ]

    try:
        numba_ver = metadata.version('numba')
    except metadata.PackageNotFoundError:
        numba_ver = 'absent'
    catalog = {
        'af_bins': _AF_BIN_META,
        'groups': group_list,
        'metadata': {
            'build_python': platform.python_version(),
            'build_timestamp': build_ts,
            'n_duplicates_collapsed': process_stats['n_dup'],
            'n_rows_raw': n_rows_raw,
            'n_skipped_invalid_base': process_stats['n_skipped_base'],
            'n_skipped_wrong_k': process_stats['n_skipped_k'],
            'numba_version': numba_ver,
            'numpy_version': np.__version__,
            'platform': platform.system() + '/' + platform.machine(),
            'source_filename': Path(in_path).name,
        },
    }
    return json.dumps(catalog, sort_keys=True, separators=(',', ':')).encode()


def _write_ndb(out_path: str, in_path: str, k: int, kmer_arr, group_arr, canonical_groups: dict,
               n_unique: int, source_crc32: int, build_ts: int,
               n_rows_raw: int, process_stats: dict, build_stats: dict):
    """Serialise arrays and metadata to .ndb format."""
    n_entries = len(kmer_arr)
    n_groups = len(canonical_groups)

    catalog_json = _build_catalog_json(in_path, canonical_groups, build_ts,
                                       n_rows_raw, process_stats, build_stats)

    catalog_size = len(catalog_json)
    pre_kmer = 64 + catalog_size
    kmer_off = ((pre_kmer + 4095) // 4096) * 4096
    kmer_bytes = n_entries * 4
    group_off = (((kmer_off + kmer_bytes) + 4095) // 4096) * 4096

    flags = 1  # bit 0: sorted

    hdr = HEADER_STRUCT.pack(
        MAGIC, 1, 0, k, 0,
        n_entries, n_groups, catalog_size, 0,
        kmer_off, group_off, flags, source_crc32, build_ts, n_unique,
    )
    if len(hdr) != 64:
        raise SystemExit(f"internal error: HEADER_STRUCT produced {len(hdr)} bytes (expected 64)")

    with open(out_path, 'wb') as f:
        f.write(hdr)
        f.write(catalog_json)
        f.write(b'\x00' * (kmer_off - 64 - catalog_size))
        f.write(kmer_arr.astype('<u4').tobytes())
        f.write(b'\x00' * (group_off - kmer_off - kmer_bytes))
        f.write(group_arr.astype('<u2').tobytes())



def _fmt_mb(n_bytes):
    mb = n_bytes / 1e6
    return f"{mb:.1f} MB" if mb < 100 else f"{mb:.0f} MB"


def _default_ndb_path(in_path: str) -> str:
    name = Path(in_path).name
    for suffix in ('.csv.tar.gz', '.csv.gz', '.csv'):
        if name.endswith(suffix):
            return str(Path(in_path).with_name(name[:-len(suffix)] + '.ndb'))
    return str(Path(in_path).with_suffix('.ndb'))


def _print_build_summary(out_path: str, k: int, n_rows_raw: int, n_skipped: int,
                         n_unique: int, n_entries: int, n_groups: int, elapsed: float):
    """Print build statistics to stderr."""
    out_size = _fmt_mb(Path(out_path).stat().st_size)
    print(f"  k:             {k}", file=sys.stderr)
    print(f"  rows read:     {n_rows_raw:,}", file=sys.stderr)
    print(f"  N-skipped:     {n_skipped:,}", file=sys.stderr)
    print(f"  unique neomers:{n_unique:,}", file=sys.stderr)
    print(f"  entries (rows):{n_entries:,}", file=sys.stderr)
    print(f"  groups:        {n_groups:,}", file=sys.stderr)
    print(f"  output:        {Path(out_path).name} ({out_size})", file=sys.stderr)
    print(f"  elapsed:       {int(elapsed // 60)}m {int(elapsed % 60)}s", file=sys.stderr)



def _run_pipeline(args):
    """Stream CSV → sort/dedup → write .ndb for dense lookup at query time."""
    in_path = args.input
    out_path = args.output or _default_ndb_path(in_path)
    t0 = time.time()

    print(f"build_neomersdb.py {_VERSION}", file=sys.stderr)
    in_size = _fmt_mb(Path(in_path).stat().st_size)
    print(f"  input:         {Path(in_path).name} ({in_size})", file=sys.stderr)

    tf, crc_reader, reader = _open_csv(in_path)
    try:
        try:
            next(reader)  # header row
            first_row = next(reader)
        except StopIteration:
            raise SystemExit(f"Error: '{Path(in_path).name}' is empty or contains only a header row")
        k = _infer_k(first_row)
        if k not in range(11, 17):
            raise SystemExit(f"Inferred k={k} is not in the supported range [11, 16]")
        arr, group_dict, proc_stats = _process_rows(reader, k, first_row=first_row)
    finally:
        tf.close()

    source_crc32 = crc_reader.crc32
    n_rows_raw = proc_stats['n_rows']

    arr, canonical_groups, n_dup = _remap_and_sort(arr, group_dict)
    proc_stats['n_dup'] = n_dup

    kmer_arr = arr['k']
    group_arr = arr['g']
    n_entries = len(arr)
    unique_kmers, kmer_counts = np.unique(kmer_arr, return_counts=True)
    n_unique = len(unique_kmers)
    n_groups = len(canonical_groups)

    n_kmers_arr, n_excl_arr = _compute_group_stats(kmer_arr, group_arr, n_groups, unique_kmers, kmer_counts)
    build_stats = {'n_kmers': n_kmers_arr, 'n_exclusive': n_excl_arr}

    build_ts = int(time.time())
    _write_ndb(out_path, in_path, k, kmer_arr, group_arr, canonical_groups,
               n_unique, source_crc32, build_ts,
               n_rows_raw, proc_stats, build_stats)

    elapsed = time.time() - t0
    if n_entries == 0:
        print("  WARNING: 0 valid entries — check source CSV and k-mer length", file=sys.stderr)
    _print_build_summary(out_path, k, n_rows_raw, proc_stats['n_skipped_base'],
                         n_unique, n_entries, n_groups, elapsed)


def main():
    ap = argparse.ArgumentParser(description='Build a .ndb index from a neomerDB CSV.tar.gz.')
    ap.add_argument('input', help='Path to neomers_K.csv.tar.gz')
    ap.add_argument('-o', '--output', help='Output .ndb path (default: stem.ndb)')
    args = ap.parse_args()

    _run_pipeline(args)


if __name__ == '__main__':
    main()
