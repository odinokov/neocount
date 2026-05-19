# neocount

> **Early prototype.** This tool is under development. APIs, output formats, and index file layouts may change without notice. Not yet validated for clinical or production use.
>
> This codebase was developed with assistance from [Claude](https://claude.ai) (Anthropic) and [Codex](https://openai.com) (OpenAI). AI-generated code may contain subtle bugs or incorrect assumptions — review outputs carefully and validate against known results before drawing biological conclusions.

`neocount` is a computational liquid-biopsy tool that scans BAM/CRAM files for cancer-associated [neomers](https://neomerdb.com) — reference-absent k-mers reintroduced by somatic mutation — and converts sparse read-level evidence into normalized pan-cancer biomarker features.

## Install

Requires **Python ≥ 3.11**.

```bash
pip install pysam numpy numba
```

`numba` accelerates the dense in-memory lookup and batched read-pair counting
path used by `count_neomers.py`.


## Build the index

Run once per neomerDB release. Pre-built index files for all k values are available on [Zenodo](https://zenodo.org/records/15518511).

```bash
python build_neomersdb.py neomersdb/neomers_13.csv.tar.gz -o neomersdb/neomers_13.ndb
python build_neomersdb.py neomersdb/neomers_15.csv.tar.gz -o neomersdb/neomers_15.ndb
```

Build time and RAM figures are theoretical estimates.

| k | Min RAM | Build time | Index size | Use case |
|---|---------|------------|------------|-----------------|
| 13 | 256 MB | < 60 s | 7 MB | Development / prototyping |
| 14 | 1.5 GB | < 5 min | 74 MB | Low-RAM machines |
| 15 | 4 GB | < 15 min | ~590 MB | Production liquid biopsy |
| 16 | 6 GB | < 60 min | ~1.45 GB | Future dense-counting target |
| 17 | >20 GB | not supported | ~8–16 GB | Requires uint64 dtype + external sort (future) |


## Query

```bash
# BAM — default normalisation: chr2q arm + chr19p arm
python count_neomers.py sample.bam --db neomersdb/neomers_15.ndb -o out.tsv

# CRAM — reference FASTA required
python count_neomers.py sample.cram --db neomersdb/neomers_15.ndb --reference GRCh38.fa -o out.tsv

# Alternative normalisation: all chr1–22 reads
python count_neomers.py sample.bam --db neomersdb/neomers_13.ndb --norm-autosomes -o out.tsv

# Tune pending-mate cache (default 5M; reduce on low-RAM machines)
python count_neomers.py sample.bam --db neomersdb/neomers_15.ndb --max-pending 1000000

# Use 1-2 BGZF decompression threads for compressed BAM/CRAM input
python count_neomers.py sample.bam --db neomersdb/neomers_15.ndb --threads 2 -o out.tsv

# Skip exact neomer strings when only group counts are needed
python count_neomers.py sample.bam --db neomersdb/neomers_15.ndb --no-neomers -o out.tsv

# Suppress progress logging
python count_neomers.py sample.bam --db neomersdb/neomers_15.ndb --log-every 0
```

`count_neomers.py` preloads the `.ndb` arrays and builds dense direct-address
lookup tables in RAM. For k=15 this is intentionally memory-heavy and suited to
hosts with about 32 GB RAM. Dense counting currently supports k≤15.

`--threads` is passed to pysam/htslib for BAM/CRAM decompression only. Use `1`
for tiny files, `2` for normal local runs, and at most `4` for large compressed
inputs. The Numba dense-counting kernels are single-threaded JIT-compiled loops,
so there is no separate Numba thread count to tune.

## Output

Clean TSV, loads with `pandas.read_csv(f, sep='\t')`:

```
sample  bam  db  k  cancer_type  organ  variant_class  af_bin  af_bin_label  pair_count  norm_region  norm_reads  neomers
```

| Column | Description |
|--------|-------------|
| `sample` | Basename prefix of the BAM/CRAM (up to the first `.`) |
| `bam` | BAM/CRAM filename |
| `db` | `.ndb` index filename |
| `k` | k-mer length |
| `cancer_type` | Cancer type label from neomerDB |
| `organ` | Organ label from neomerDB |
| `variant_class` | Somatic variant class (e.g. `Missense_Mutation`, `IGR`) |
| `af_bin` | Germline AF bin index (0–5) |
| `af_bin_label` | Human-readable AF range (`<0.001`, `0.001-0.01`, …, `Unknown`) |
| `pair_count` | Read pairs where neomers from this group appear in **both** mates |
| `norm_region` | Normalisation mode: `chr2q+chr19p` or `autosomes` |
| `norm_reads` | Total filtered reads in the normalization region |
| `neomers` | Comma-separated list of exact neomer sequences matched in this group |

Only groups with `pair_count > 0` are emitted. The primary downstream signal is:

```python
df['signal'] = df['pair_count'] / df['norm_reads']
```

Use the ratio for cross-sample comparison; raw `pair_count` scales with library size and k.

Groups aggregate neomers by `(cancer_type, organ, variant_class, af_bin)`. To filter for high-confidence coding hits:

```python
coding = df[df['variant_class'].isin(['Missense_Mutation', 'Nonsense_Mutation', 'Splice_Site'])]
specific = df[df['af_bin'] == 0]   # AF < 0.001, lowest germline contamination risk
```

A run summary is printed to stderr on completion:

```
v0.1  db=neomers_15.ndb  k=15  norm=chr2q+chr19p
AND logic: group counted only when neomers found in BOTH mates
reads=250,000,000  pairs=124,800,000  tlen_filtered=14,200,000  evicted=0 (0.0%)  norm_reads=18,500,000  elapsed=312.4s
```

`tlen_filtered` counts reads or pairs discarded because `|TLEN| > 2·read_length − k`; these fragments are too long for k-mer co-visibility between mates. This includes first-seen reads rejected conservatively before their mate is encountered.

## Normalisation

The default normalisation region is the **chr2q arm + chr19p arm** (`chr2:95000000-242000000` and `chr19:0-24200000` for both hg19 and hg38). Both arms are largely copy-neutral across pan-cancer analyses (Hartwig et al., *eLife* 2019), making read depth in these regions a stable proxy for input DNA quantity.

Use `--norm-autosomes` to count all filtered reads on chr1–22 instead (higher read count, but more susceptible to copy-number variation in the sample).


## Known limitations

- **No mate-overlap consensus.** AND logic requires *some* neomer from the
  same group in each mate; it does not require the **same** neomer at the
  **same** fragment position.


## Reference

If you use neomerDB in your work, please cite:

Provatas, K., Chan, C. S. Y., Kerasiotis, I., Bochalis, E., Nayak, A., Zacharia, B. E., Pavlopoulos, G. A., Li, W., & Georgakopoulos-Soares, I. (2026). neomerDB: A comprehensive database of neomer biomarkers in cancer. _Database, 2026_, baag006. https://doi.org/10.1093/database/baag006
