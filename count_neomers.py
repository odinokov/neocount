#!/usr/bin/env python3
"""count_neomers v0.1 — neomer detection from BAM/CRAM."""

import argparse
import csv
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pysam

from ndb import NmerDB, process_pair_batch_dense, decode_kmer, AF_BIN_LABELS

_VERSION = 'v0.1'
_EXCLUDE_FLAGS = 4 | 256 | 512 | 1024 | 2048  # unmapped, secondary, qcfail, dup, supplementary
_DEFAULT_NORM_REGIONS = (
    'chr2:95000000-242000000,'
    '2:95000000-242000000,'
    'chr19:0-24200000,'
    '19:0-24200000'
)
_MAX_EPOCH = int(np.iinfo(np.uint32).max) - 1


def _parse_norm_regions(regions_str: str) -> dict:
    result = {}
    for reg in regions_str.split(','):
        chrom, coords = reg.split(':')
        start, end = coords.split('-')
        result[chrom] = (int(start), int(end))
    return result


def _build_autosome_set() -> set:
    return {str(i) for i in range(1, 23)} | {f'chr{i}' for i in range(1, 23)}


def _in_norm_region(read, norm_dict: dict, autosome_set, use_autosomes: bool) -> bool:
    chrom = read.reference_name
    if chrom is None:
        return False
    if use_autosomes:
        return chrom in autosome_set
    if chrom not in norm_dict:
        return False
    start, end = norm_dict[chrom]
    return start <= read.reference_start < end


def _enqueue_read(pending: OrderedDict, qname: str, seq: str, max_pending: int) -> int:
    """Add read to pending cache; evict oldest if at capacity. Returns number of evictions (0 or 1)."""
    evicted = 0
    if len(pending) >= max_pending:
        pending.popitem(last=False)
        evicted = 1
    pending[qname] = seq
    return evicted


def _validate_args(ap, args) -> None:
    if args.min_mapq < 0:
        ap.error("--min-mapq must be >= 0")
    if args.max_pending < 1:
        ap.error("--max-pending must be >= 1")
    if args.batch_size < 1:
        ap.error("--batch-size must be >= 1")
    if args.batch_read_width < 1:
        ap.error("--batch-read-width must be >= 1")
    if args.threads < 1:
        ap.error("--threads must be >= 1")
    if args.threads > 4:
        print("WARNING: --threads is for BAM/CRAM BGZF decompression; "
              "capping to 4", file=sys.stderr)
        args.threads = 4


def _is_usable_read(read, min_mapq: int) -> bool:
    return not (read.flag & _EXCLUDE_FLAGS or read.mapping_quality < min_mapq)


def _tlen_ok(tlen: int, read_len: int, k: int) -> bool:
    """Return True when the fragment allows k-mer co-visibility in both mates.

    Co-visibility requires |TLEN| ≤ 2·read_length − k.  TLEN=0 (unmapped
    mate or different-chromosome pair) is passed through unchecked.
    """
    return tlen == 0 or tlen <= 2 * read_len - k


def _alloc_pair_batch(batch_size: int, read_width: int):
    seq1 = np.empty((batch_size, read_width), dtype=np.uint8)
    seq2 = np.empty((batch_size, read_width), dtype=np.uint8)
    len1 = np.empty(batch_size, dtype=np.uint16)
    len2 = np.empty(batch_size, dtype=np.uint16)
    return seq1, seq2, len1, len2


def _next_batch_width(current: int, needed: int) -> int:
    width = current
    while width < needed:
        width *= 2
    return width


def _store_seq(dst: np.ndarray, seq: str) -> int:
    data = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    if len(data) > np.iinfo(np.uint16).max:
        raise SystemExit(f"read length {len(data)} exceeds uint16 batch length storage")
    dst[:len(data)] = data
    return len(data)


def _flush_pair_batch(n_batch: int, seq1_batch: np.ndarray, len1_batch: np.ndarray,
                      seq2_batch: np.ndarray, len2_batch: np.ndarray, db: NmerDB,
                      group_counts: np.ndarray, observed_entries: np.ndarray,
                      seen_group: np.ndarray, counted_group: np.ndarray,
                      epoch: int, emit_neomers: bool) -> int:
    if n_batch == 0:
        return epoch
    if epoch + n_batch >= _MAX_EPOCH:
        seen_group.fill(0)
        counted_group.fill(0)
        epoch = 1
    return process_pair_batch_dense(
        seq1_batch, len1_batch, seq2_batch, len2_batch, n_batch, db,
        group_counts, observed_entries, seen_group, counted_group,
        epoch, emit_neomers,
    )


def _log_progress(n_reads: int, n_pairs: int, reads_last: int, t_last: float, log_every: int):
    if log_every <= 0 or n_reads % log_every != 0:
        return t_last, reads_last
    now = time.time()
    speed = (n_reads - reads_last) / max(now - t_last, 1e-9)
    print(f"[{n_reads/1e6:.1f}M reads, {n_pairs/1e6:.1f}M pairs, "
          f"{speed/1e6:.2f}M reads/s]", file=sys.stderr, flush=True)
    return now, n_reads


def _scan_reads(bam_path: str, db: NmerDB, args):
    """One-pass coordinate-sorted read scan with batched dense Numba counting."""
    open_kwargs = {'reference_filename': args.reference} if args.reference else {}
    if args.threads > 1:
        open_kwargs['threads'] = args.threads

    norm_dict = _parse_norm_regions(_DEFAULT_NORM_REGIONS)
    autosome_set = _build_autosome_set()
    use_autosomes = args.norm_autosomes

    n_groups = max(db.catalog) + 1 if db.catalog else 0
    pending = OrderedDict()
    group_counts = np.zeros(n_groups, dtype=np.uint64)
    observed_entries = np.zeros(len(db.kmer_arr), dtype=np.bool_)
    seen_group = np.zeros(n_groups, dtype=np.uint32)
    counted_group = np.zeros(n_groups, dtype=np.uint32)
    read_width = args.batch_read_width
    seq1_batch, seq2_batch, len1_batch, len2_batch = _alloc_pair_batch(args.batch_size, read_width)
    n_batch = 0
    epoch = 1
    n_reads = n_pairs = n_evicted = n_tlen_filtered = n_norm = 0
    t0 = t_last = time.time()
    reads_last = 0
    log_every = args.log_every

    with pysam.AlignmentFile(bam_path, 'rb', **open_kwargs) as bam:
        for read in bam:
            if not _is_usable_read(read, args.min_mapq):
                continue
            n_reads += 1

            if _in_norm_region(read, norm_dict, autosome_set, use_autosomes):
                n_norm += 1

            t_last, reads_last = _log_progress(n_reads, n_pairs, reads_last, t_last, log_every)

            if not read.is_paired:
                continue

            seq = read.query_sequence
            qname = read.query_name
            if seq is None or qname is None:
                continue
            seq1 = pending.pop(qname, None)
            if seq1 is not None:
                if _tlen_ok(abs(read.template_length), min(len(seq), len(seq1)), db.k):
                    needed_width = max(len(seq1), len(seq))
                    if needed_width > read_width:
                        epoch = _flush_pair_batch(
                            n_batch, seq1_batch, len1_batch, seq2_batch, len2_batch, db,
                            group_counts, observed_entries, seen_group, counted_group,
                            epoch, args.emit_neomers,
                        )
                        n_batch = 0
                        read_width = _next_batch_width(read_width, needed_width)
                        seq1_batch, seq2_batch, len1_batch, len2_batch = _alloc_pair_batch(
                            args.batch_size, read_width,
                        )
                    if n_batch == args.batch_size:
                        epoch = _flush_pair_batch(
                            n_batch, seq1_batch, len1_batch, seq2_batch, len2_batch, db,
                            group_counts, observed_entries, seen_group, counted_group,
                            epoch, args.emit_neomers,
                        )
                        n_batch = 0
                    len1_batch[n_batch] = _store_seq(seq1_batch[n_batch], seq1)
                    len2_batch[n_batch] = _store_seq(seq2_batch[n_batch], seq)
                    n_batch += 1
                    n_pairs += 1
                else:
                    n_tlen_filtered += 1
            else:
                tlen = abs(read.template_length)
                if tlen and tlen > 2 * len(seq) - db.k:
                    n_tlen_filtered += 1
                else:
                    n_evicted += _enqueue_read(pending, qname, seq, args.max_pending)

    epoch = _flush_pair_batch(
        n_batch, seq1_batch, len1_batch, seq2_batch, len2_batch, db,
        group_counts, observed_entries, seen_group, counted_group,
        epoch, args.emit_neomers,
    )
    elapsed = time.time() - t0
    stats = dict(n_reads=n_reads, n_pairs=n_pairs, n_evicted=n_evicted,
                 n_tlen_filtered=n_tlen_filtered, n_norm=n_norm,
                 elapsed=elapsed)
    return group_counts, n_norm, stats, observed_entries


def _collect_observed_neomers(db: NmerDB, observed_entries: np.ndarray | None) -> dict:
    if observed_entries is None:
        return {}
    result: dict = {}
    for idx in np.flatnonzero(observed_entries):
        gid = int(db.group_arr[idx])
        result.setdefault(gid, set()).add(int(db.kmer_arr[idx]))
    return result


def write_output(group_counts: np.ndarray, norm_reads: int, db: NmerDB, args, out_fh,
                 observed_entries: np.ndarray | None = None):
    cols = ['sample', 'bam', 'db', 'k', 'cancer_type', 'organ', 'variant_class',
            'af_bin', 'af_bin_label', 'pair_count', 'norm_region', 'norm_reads', 'neomers']

    bam_base = os.path.basename(args.bam)
    sample = bam_base.split('.', 1)[0] or bam_base
    db_name = os.path.basename(args.db)
    norm_label = 'autosomes' if args.norm_autosomes else 'chr2q+chr19p'
    k = db.k
    observed_by_group = _collect_observed_neomers(db, observed_entries) if args.emit_neomers else {}

    writer = csv.writer(out_fh, delimiter='\t', lineterminator='\n')
    writer.writerow(cols)
    for gid in np.flatnonzero(group_counts):
        gid = int(gid)
        g = db.catalog[gid]
        af_bin = g['af_bin']
        kmers = observed_by_group.get(gid, set())
        neomer_str = ','.join(sorted(decode_kmer(v, k) for v in kmers))
        writer.writerow([
            sample, bam_base, db_name, k,
            g['cancer_type'], g['organ'], g['variant_class'],
            af_bin, AF_BIN_LABELS[af_bin],
            int(group_counts[gid]), norm_label, norm_reads, neomer_str,
        ])


def main():
    ap = argparse.ArgumentParser(
        description='Count neomer pairs in a BAM/CRAM file against a .ndb index.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('bam', help='BAM or CRAM input file')
    ap.add_argument('--db', required=True, help='Path to .ndb index (required)')
    ap.add_argument('--reference', '-T', help='Reference FASTA (required for CRAM)')
    ap.add_argument('-o', '--output', default='-', help='Output TSV path (- for stdout)')
    ap.add_argument('--norm-autosomes', action='store_true',
                    help='Normalise to all chr1–22 reads instead of chr2q+chr19p')
    ap.add_argument('--max-pending', type=int, default=5_000_000,
                    help='Max pending-mate cache size (reduce on low-RAM machines)')
    ap.add_argument('--batch-size', type=int, default=100_000,
                    help='Read pairs per Numba dense-lookup batch')
    ap.add_argument('--batch-read-width', type=int, default=256,
                    help='Initial per-read batch width; grows automatically for longer reads')
    ap.add_argument('--no-neomers', action='store_false', dest='emit_neomers',
                    help='Skip exact neomer strings in output for faster counting')
    ap.add_argument('--no-preload-index', action='store_true',
                    help='Keep index arrays mmap-backed instead of copying them into RAM')
    ap.add_argument('--log-every', type=int, default=5_000_000,
                    help='Log progress every N filtered reads (0 to suppress)')
    ap.add_argument('-q', '--min-mapq', type=int, default=5, dest='min_mapq',
                    help='Minimum MAPQ (reads below this threshold are excluded)')
    ap.add_argument('--threads', type=int, default=2,
                    help='pysam BGZF decompression threads for BAM/CRAM input (1-4)')
    args = ap.parse_args()
    _validate_args(ap, args)

    with NmerDB(args.db, preload=not args.no_preload_index, dense=True) as db:
        print(
            f"dense_lookup={db.lookup_bytes / 1e9:.2f} GB"
            f"  preload={'no' if args.no_preload_index else 'yes'}"
            f"  batch_size={args.batch_size:,}"
            f"  threads={args.threads}",
            file=sys.stderr,
        )
        group_counts, norm_reads, stats, observed_entries = _scan_reads(args.bam, db, args)

        if args.output == '-':
            write_output(group_counts, norm_reads, db, args, sys.stdout, observed_entries)
        else:
            with open(args.output, 'w') as f:
                write_output(group_counts, norm_reads, db, args, f, observed_entries)

    n = stats
    pct_evict = 100 * n['n_evicted'] / max(n['n_reads'], 1)

    if n['n_pairs'] == 0 and n['n_reads'] > 1000:
        print("WARNING: 0 pairs formed; input may be single-end or unsorted — "
              "requires paired-end reads for neomer counting", file=sys.stderr)
    if n['n_reads'] > 0 and n['n_evicted'] / n['n_reads'] > 0.01:
        print(f"WARNING: {pct_evict:.1f}% of filtered reads evicted from pending cache; "
              "increase --max-pending", file=sys.stderr)

    print(
        f"\n{_VERSION}  db={os.path.basename(args.db)}  k={db.k}"
        f"  norm={'autosomes' if args.norm_autosomes else 'chr2q+chr19p'}"
        f"\nAND logic: group counted only when neomers found in BOTH mates"
        f"\nreads={n['n_reads']:,}  pairs={n['n_pairs']:,}  "
        f"tlen_filtered={n['n_tlen_filtered']:,}  "
        f"evicted={n['n_evicted']:,} ({pct_evict:.1f}%)  "
        f"norm_reads={n['n_norm']:,}  elapsed={n['elapsed']:.1f}s",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
