# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact Python CLI project for building and querying neomerDB indexes.

- `build_neomersdb.py` builds `.ndb` binary indexes from `neomers_K.csv.tar.gz` inputs.
- `count_neomers.py` scans BAM/CRAM files against an `.ndb` index and writes TSV output.
- `ndb.py` contains the shared `.ndb` file format, metadata, and array loading logic.
- `README.md` is the user-facing source of truth for installation, command examples, output columns, and limitations.
- Generated data such as `.ndb`, BAM/CRAM, TSV reports, and large neomerDB downloads should stay out of git unless explicitly requested.

## Build, Test, and Development Commands

- `pip install pysam numpy numba` installs the runtime dependencies documented for local development.
- `python3 build_neomersdb.py neomersdb/neomers_13.csv.tar.gz -o neomersdb/neomers_13.ndb` builds a small development index.
- `python3 count_neomers.py sample.bam --db neomersdb/neomers_13.ndb -o out.tsv` runs a local count job.
- `python3 -m py_compile build_neomersdb.py count_neomers.py ndb.py` catches syntax errors without requiring test data.
- `python3 count_neomers.py --help` and `python3 build_neomersdb.py --help` verify CLI argument wiring after edits.

## Coding Style & Naming Conventions

Use Python 3.11-compatible code, 4-space indentation, and descriptive snake_case names for functions, variables, and CLI-derived values. Keep constants uppercase when they represent stable format or configuration values. Prefer explicit argparse help text when behavior changes, and keep README examples aligned with CLI defaults. Avoid broad rewrites in the hot counting path unless the change is measured or needed for correctness.

## Testing Guidelines

There is no dedicated test suite in this checkout. For every change, run `python3 -m py_compile ...` on all three Python files. For CLI changes, run the relevant `--help` command and, when data is available, a small k=13 smoke run. Compare output TSVs with `diff -u` or checksums when changing counting, normalization, or `.ndb` loading behavior.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, sometimes prefixed with a release marker, for example `Drop batch width from count summary` or `v0.1: update CLI release marker`. Keep commits focused and mention affected scripts. Pull requests should describe the behavioral change, list validation commands and sample data used, and call out any output-format, index-format, performance, or README updates.

## Agent-Specific Instructions

Preserve the distinction between user-facing release strings and `.ndb` file-format compatibility. Do not change binary format guards in `ndb.py` unless the index layout is intentionally changing.
