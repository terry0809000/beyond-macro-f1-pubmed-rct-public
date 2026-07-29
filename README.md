# Beyond Macro-F1: PubMed RCT sentence classification

Reproducibility code and sanitized derived evidence for:

> **Beyond Macro-F1: Do Higher-Performing Biomedical NLP Models Show Better
> Calibration and Perturbation Consistency? A Benchmark Study of PubMed RCT
> Sentence Classification**

This history-free repository candidate accompanies the verified PLOS ONE
revision run. It is deliberately smaller than the archival deposit: GitHub
contains code, provenance, figures, and compact result tables, while the full
sanitized evidence archive is intended for Zenodo.

## Release identity

- Repository: `terry0809000/beyond-macro-f1-pubmed-rct-public`
- Version: `v1.0.0`
- Run ID: `PLOS_R1_publication_20260729T084516Z`
- Source release:
  `PLOS_R1_REVIEWER_ALIGNED_G4_FINAL_EXPORT_COMPLETE_CLASSICAL_PARAMS_20260729`
- Certified source notebook SHA-256:
  `f41a9b1542f4a544eea2ff2085cc096743eb6f98d166a6e2dc0af4e96590ba13`
- Completion contract: `plos_publication_completion_v3`
- Pre-license sanitized evidence archive:
  `PLOS_R1_publication_20260729T084516Z_sanitized_data_code_deposit.zip`
- Archive bytes: `64,452,428`
- Archive SHA-256:
  `8d648613dde0134c828faab507634481a827e22ce16fbfc9e7dd057a03ee0f63`
- Zenodo DOI: pending creation from Zenodo's archive of the GitHub `v1.0.0`
  release. Insert only the identifier actually generated for that release
  after it resolves and the archive is independently read back.

The bound archive above preserves the sanitized analytical payload produced
before the author approved the licenses. It does not contain the three license
files now in this Git staging tree. A public Zenodo payload should be rebuilt
to include `LICENSE`, `LICENSE-CONTENT-DATA-CC-BY-4.0.txt`, and `LICENSES.md`;
its new byte count and SHA-256 must then replace the pre-license values above.

The publication run completed with 16 final system/seed rows, 10,000
complete-abstract bootstrap replicates per reported contrast, a locked
pre-test comparison plan, and a verified G4 runtime. The completion sentinel
and authoritative 83-artifact manifest are retained under `provenance/`.

## Verify this GitHub release

The lightweight verifier uses only the Python standard library.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m plos_nlp_release .
python -m pytest -q
```

It checks:

- every staged file against `release_manifest.csv` by path, byte count, and
  SHA-256;
- the exact run, release, source-notebook hash, and completion contract;
- selected final-test, bootstrap, grouped-ablation, factorial, calibration,
  perturbation, per-class, and efficiency schemas;
- absence of high-confidence credentials, personal user paths, raw benchmark
  text columns, model weights, licensed fonts, and retired manual-review data;
- the GitHub file-size ceiling used for this staging tree.

## Repository layout

```text
analysis/
  certified_source_notebook.ipynb  exact hash-bound publication runner
  run_smoke_analysis.py            portable, path-cleaned smoke extraction
  PLOS_COLAB_RUNTIME_CONFIG_PUBLIC_TEMPLATE.json
artifacts/summary/                 compact, no-text result and audit tables
figures/                           canonical PDF/PNG exports and source CSVs
provenance/                        completion sentinel and source manifests
scripts/                           upstream normalization and data verification
src/plos_nlp_release/              standard-library release verifier
tests/                             verifier and result-contract smoke tests
WITHHELD_ARTIFACTS.csv             explicit non-Git inventory
release_manifest.csv               Git-tree checksums
```

## Obtain and verify the benchmark

Raw PubMed RCT text is third-party material and is not redistributed here.
Obtain the upstream `PubMed_20k_RCT` split files from the immutable commit:

<https://github.com/Franck-Dernoncourt/pubmed-rct/tree/17ed2cb0590decfca0266add0c76f254f67232b4/PubMed_20k_RCT>

Normalize each upstream split into the three-column format:

```bash
python scripts/normalize_official_pubmed_rct.py UPSTREAM/train.txt data/train.txt
python scripts/normalize_official_pubmed_rct.py UPSTREAM/dev.txt data/dev.txt
python scripts/normalize_official_pubmed_rct.py UPSTREAM/test.txt data/test.txt
python scripts/verify_authoritative_pubmed_rct.py data
```

The verifier requires these normalized SHA-256 values:

| Split | Rows | Abstracts | SHA-256 |
|---|---:|---:|---|
| train | 180,040 | 15,000 | `2b1be3797844d060873d97e445aa4bf05fafbd050d526528364e233c17a7ced0` |
| dev | 30,212 | 2,500 | `21019fb5fde4ed4fd67a358ae2a74aadf3b85540aa42ef2fb41d7a86d7aeda82` |
| test | 30,135 | 2,500 | `b5dea844fecf520933f67f5dcd216c6cf1250f92946170cd93e55302fbb07d50` |

Do not commit these raw or normalized text files.

## Run the analysis

### Portable smoke run

The path-cleaned Python extraction uses the notebook's reduced-data integration
profile and cannot create an authoritative publication sentinel. It still
exercises the transformer path and therefore requires exactly one
BF16-capable CUDA GPU; it is not a CPU-only example.

```bash
python -m pip install -e ".[analysis]"
export PLOS_DATA_DIR="$PWD/data"
export PLOS_OUTPUT_DIR="$PWD/outputs"
export PLOS_FONT_PATH="/path/to/a/legitimately-licensed/Arial-file"
python analysis/run_smoke_analysis.py
```

PowerShell equivalents use `$env:PLOS_DATA_DIR`, `$env:PLOS_OUTPUT_DIR`, and
`$env:PLOS_FONT_PATH`.

### Exact publication workflow

Use `analysis/certified_source_notebook.ipynb` for a full rerun. It is copied
byte-for-byte from the verified run and its SHA-256 must equal the value in
the release identity above. Configure the notebook with
`analysis/PLOS_COLAB_RUNTIME_CONFIG_PUBLIC_TEMPLATE.json`, the verified
upstream data, and a legitimately licensed Arial file supplied privately via
`PLOS_FONT_PATH`. The repository does not include a font file or a private
font path.

The exact run was executed on an NVIDIA RTX PRO 6000 Blackwell Server Edition
G4 runtime. A full rerun is computationally intensive; the smoke runner is the
appropriate installation check.

## GitHub versus Zenodo contents

GitHub intentionally omits large row-level prediction, out-of-fold,
probability, and bootstrap-distribution files. The largest excluded file is
`fixed_ablation_oof_predictions_long.csv` (125,892,851 bytes), which exceeds
the conservative 90 MiB Git staging limit. `WITHHELD_ARTIFACTS.csv` records
every excluded source-deposit file and the reason.

The complete sanitized evidence archive is the basis of the Zenodo payload.
Before publication:

1. rebuild the archive with the complete approved license files and verify its
   new filename, byte count, and SHA-256;
2. apply the approved MIT code license and CC BY 4.0 content/data license in
   the Zenodo metadata and description;
3. confirm that Zenodo's GitHub integration is enabled for this public
   repository;
4. publish GitHub release `v1.0.0` and allow Zenodo to archive that release;
5. independently download/read back the archived payload and verify its byte
   count and SHA-256;
6. verify the generated version and concept DOI(s) in a signed-out browser;
7. update the manuscript Data Availability Statement, `CITATION.cff`, and this
   README with the verified identifier.

See `ZENODO_DEPOSIT_CHECKLIST.md`.

## Data and privacy boundary

Included CSVs contain aggregate metrics, class-level summaries, figure source
data, system specifications, and public run metadata. They contain no abstract
sentence text. Item-level no-text derivatives and bulky resampling
distributions remain in the complete sanitized archive rather than Git.

No credentials, model weights, licensed font files, raw PubMed RCT split
files, or rejected manual/clinician-review datasets are distributed. Mentions
of excluded workflows in the certified notebook are provenance statements,
not included annotation material.

## Licenses

Copyright © 2026 Tianyi Yu.

- Author-generated code is licensed under the
  [MIT License](LICENSE).
- Author-generated documentation, figures, figure source data, and derived
  outputs are licensed under
  [Creative Commons Attribution 4.0 International](LICENSE-CONTENT-DATA-CC-BY-4.0.txt).

`LICENSES.md` defines the file-level boundary and attribution request.
These grants do not relicense third-party PubMed RCT text, pretrained models,
libraries, model weights, licensed fonts, names, or trademarks. Raw PubMed RCT
text remains absent and must be obtained from the pinned upstream source under
its own terms.

## Citation

Versioned software citation metadata are in `CITATION.cff`. No DOI is claimed
before Zenodo archives GitHub release `v1.0.0`; the generated DOI will be
inserted only after it resolves publicly and the archive is read back
successfully.

## Reproducibility scope

The compact GitHub tables are sufficient to audit reported summaries and
figures, but they are not a substitute for the complete sanitized Zenodo
archive or the third-party input data. Model-download availability, GPU
hardware, and future package compatibility can change; immutable upstream
commits, checkpoint revisions, package constraints, and the certified
notebook are therefore recorded explicitly.
