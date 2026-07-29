# Beyond Macro-F1: PubMed RCT sentence classification

Reproducibility code and sanitized derived evidence for:

> **Beyond Macro-F1: Do Higher-Performing Biomedical NLP Models Show Better
> Calibration and Perturbation Consistency? A Benchmark Study of PubMed RCT
> Sentence Classification**

This history-free repository accompanies the verified PLOS ONE revision run.
The Git tree contains code, provenance, figures, and compact result tables.
The full sanitized evidence ZIP is distributed separately as a GitHub release
asset because it is intentionally not committed to the repository.

## Release identity

- Repository: `terry0809000/beyond-macro-f1-pubmed-rct-public`
- Version: `v1.0.0`
- Run ID: `PLOS_R1_publication_20260729T084516Z`
- Source release:
  `PLOS_R1_REVIEWER_ALIGNED_G4_FINAL_EXPORT_COMPLETE_CLASSICAL_PARAMS_20260729`
- Certified source notebook SHA-256:
  `f41a9b1542f4a544eea2ff2085cc096743eb6f98d166a6e2dc0af4e96590ba13`
- Completion contract: `plos_publication_completion_v3`
- Canonical license-inclusive sanitized evidence release asset:
  `PLOS_R1_publication_20260729T084516Z_sanitized_data_code_deposit.zip`
- Archive bytes: `64,461,196`
- Archive SHA-256:
  `b6f4a158891784ea5f6f451d33d4e3a46ff42406afa582553a01f2fa381fddf4`
- Public deposit manifest: `public_deposit_manifest.csv`
- Public deposit manifest bytes: `76,452`
- Public deposit manifest SHA-256:
  `27fdd8bec132208474377209f128f138f0e71fd28d111d554304dfbf79af7e37`
- Outer checksums file: `SHA256SUMS`
- Outer checksums bytes: `229`
- Outer checksums SHA-256:
  `6f160158acba129753954b5aa2caff217c24f627a977d40611181039db8be56a`
- Zenodo DOI: pending creation from Zenodo's archive of the GitHub `v1.0.0`
  release. Insert only the identifier actually generated for that archived
  tagged repository after it resolves.

The canonical ZIP contains 119 members, including `LICENSE`,
`LICENSE-CONTENT-DATA-CC-BY-4.0.txt`, and `LICENSES.md`. Its deterministic
rebuild, internal manifest/checksums, and security boundary were independently
verified. The ZIP is a manually attached GitHub release asset. Zenodo's GitHub
integration archives the tagged repository; do not assume that it also copies
manually attached GitHub release assets. Inspect the eventual Zenodo record
before describing the full ZIP as present there.

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

## Git repository, release assets, and Zenodo

GitHub intentionally omits large row-level prediction, out-of-fold,
probability, and bootstrap-distribution files. The largest excluded file is
`fixed_ablation_oof_predictions_long.csv` (125,892,851 bytes), which exceeds
the conservative 90 MiB Git staging limit. `WITHHELD_ARTIFACTS.csv` records
every excluded source-deposit file and the reason.

The complete sanitized evidence ZIP and its companion
`public_deposit_manifest.csv` and `SHA256SUMS` files are canonical GitHub
release assets. Before publication:

1. attach the exact verified ZIP, public manifest, and outer checksum file to
   GitHub release `v1.0.0`;
2. independently download the GitHub release assets and verify each byte count
   and SHA-256 shown above;
3. confirm that Zenodo's GitHub integration is enabled, then publish the
   reviewed tag/release so Zenodo archives the tagged repository;
4. inspect the generated Zenodo record and its files rather than assuming it
   contains the manually attached GitHub release asset;
5. verify the generated version and concept DOI(s) in a signed-out browser;
6. update the manuscript Data Availability Statement, `CITATION.cff`, and this
   README with only the verified identifier and observed archive contents.

See `ZENODO_DEPOSIT_CHECKLIST.md`.

## Data and privacy boundary

Included CSVs contain aggregate metrics, class-level summaries, figure source
data, system specifications, and public run metadata. They contain no abstract
sentence text. Item-level no-text derivatives and bulky resampling
distributions remain in the complete sanitized GitHub release archive rather
than the Git tree.

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
inserted only after it resolves publicly and the Zenodo record contents are
inspected.

## Reproducibility scope

The compact Git tables are sufficient to audit reported summaries and figures,
but they are not a substitute for the complete sanitized GitHub release
archive or the third-party input data. Whether the eventual Zenodo record also
contains that manually attached ZIP must be verified from the record itself.
Model-download availability, GPU hardware, and future package compatibility
can change; immutable upstream commits, checkpoint revisions, package
constraints, and the certified notebook are therefore recorded explicitly.
