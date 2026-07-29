# GitHub release and Zenodo archival checklist

Do not invent or prefill a DOI. Zenodo's GitHub integration is configured to
archive the first public `v1.0.0` release and mint its DOI. Record only the
identifier actually generated from that archived tagged repository, after it
resolves without authentication and the Zenodo record contents have been
inspected.

## Canonical GitHub release assets

- Filename:
  `PLOS_R1_publication_20260729T084516Z_sanitized_data_code_deposit.zip`
- Bytes: `64,461,196`
- SHA-256:
  `b6f4a158891784ea5f6f451d33d4e3a46ff42406afa582553a01f2fa381fddf4`
- Manifest: `public_deposit_manifest.csv`
- Manifest bytes: `76,452`
- Manifest SHA-256:
  `27fdd8bec132208474377209f128f138f0e71fd28d111d554304dfbf79af7e37`
- Outer checksums: `SHA256SUMS`
- Outer checksums bytes: `229`
- Outer checksums SHA-256:
  `6f160158acba129753954b5aa2caff217c24f627a977d40611181039db8be56a`
- Related GitHub release:
  `https://github.com/terry0809000/beyond-macro-f1-pubmed-rct-public/releases/tag/v1.0.0`

The ZIP is deliberately not committed to Git. It is the verified,
license-inclusive GitHub release asset: 119 ZIP members, including the exact
`LICENSE`, `LICENSE-CONTENT-DATA-CC-BY-4.0.txt`, and `LICENSES.md` files. Its
deterministic second build, manifest, internal checksums, and security scans
passed.

Zenodo's GitHub integration archives the tagged repository. Do not assume it
also contains this manually attached GitHub release asset unless the generated
Zenodo record is inspected and that file is observed there, or the asset is
separately uploaded and independently verified.

## Metadata

- Title: *Beyond Macro-F1: PubMed RCT sentence-classification reproducibility package*
- Creator: Tianyi Yu, King's College London
- Version: `1.0.0`
- Related manuscript: *Beyond Macro-F1: Do Higher-Performing Biomedical NLP
  Models Show Better Calibration and Perturbation Consistency? A Benchmark
  Study of PubMed RCT Sentence Classification*
- Keywords: biomedical NLP; sentence classification; PubMed RCT; calibration;
  perturbation consistency; reproducibility
- Description: identify the exact run ID, source release, source-notebook
  SHA-256, and the separation between sanitized derived evidence and
  third-party raw text.

Apply the author-approved licensing boundary:

- MIT for author-generated code; and
- CC BY 4.0 for author-generated documentation, figures, figure source data,
  and derived outputs.

If Zenodo presents a single primary-license field, describe the mixed-license
boundary prominently in the record description. The tagged repository and
canonical GitHub release ZIP both include `LICENSE`,
`LICENSE-CONTENT-DATA-CC-BY-4.0.txt`, and `LICENSES.md`. Do not imply that
either license applies to excluded third-party PubMed text, pretrained models,
libraries, fonts, or model weights.

## Draft/reviewer workflow

1. Confirm that this public repository is enabled in Zenodo's GitHub
   integration.
2. Attach the exact verified ZIP, `public_deposit_manifest.csv`, and
   `SHA256SUMS` to GitHub release `v1.0.0`.
3. Download the GitHub release assets and verify every byte count and SHA-256.
4. Publish the reviewed tag/release so Zenodo archives the tagged repository.
5. If peer review requires access before final publication, use only a
   reviewer-access mechanism whose link is tested in a signed-out browser.
   Do not commit a token-bearing preview URL to Git.
6. Keep the DOI and manuscript language pending until Zenodo has generated the
   identifier from the archived release.

## Publication verification

1. Confirm that the approved MIT/CC BY 4.0 boundary and third-party exclusions
   are visible on GitHub and in the generated Zenodo record.
2. Independently download the three GitHub release assets and confirm their
   exact byte counts and SHA-256 values.
3. Inspect every file in the Zenodo record. Do not infer that Zenodo copied the
   manually attached GitHub release ZIP.
4. If the full ZIP is separately placed in Zenodo, independently download it
   from Zenodo and confirm the same byte count and SHA-256.
5. Resolve the generated version and concept DOI(s) in a signed-out browser.
6. Add the verified DOI to `README.md`, `CITATION.cff`, and the manuscript Data
   Availability Statement.
7. Record the immutable Git commit and release tag related to the deposit.
8. Rerun the repository verifier and preserve its JSON output outside Git.
