# GitHub release and Zenodo archival checklist

Zenodo archived the public `v1.0.0` tagged repository. The identifiers and
record below were observed directly and resolved without authentication:

- Version DOI: <https://doi.org/10.5281/zenodo.21693669>
- Concept DOI: <https://doi.org/10.5281/zenodo.21693668>
- Record: <https://zenodo.org/records/21693669>
- Resolution: HTTP 200 to the record above

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

The full ZIP was independently downloaded from the GitHub release and matched
the byte count and SHA-256 above.

## Observed Zenodo record

- Access: open
- Resource type: `Software`
- Version: `v1.0.0`
- Primary license: `MIT`
- Number of files: `1`
- File:
  `terry0809000/beyond-macro-f1-pubmed-rct-public-v1.0.0.zip`
- Bytes: `2,014,751`
- MD5: `01e5fbfcdfc438cc481f03b188365508`
- Downloaded SHA-256:
  `9d84b61d2f6a133bfb4ca84bb28f5e51a231cb1243db9fef30d98cb92179b780`
- Extracted verification: repository verifier passed all `86` manifest rows;
  `4` tests passed.

The Zenodo record contains exactly the generated tagged-repository ZIP above.
It does not contain the separate 64,461,196-byte manually attached GitHub
evidence asset.

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

Zenodo exposes `MIT` as the record's primary software license. The tagged
repository and canonical GitHub release ZIP both include `LICENSE`,
`LICENSE-CONTENT-DATA-CC-BY-4.0.txt`, and `LICENSES.md`. Do not imply that
either license applies to excluded third-party PubMed text, pretrained models,
libraries, fonts, or model weights.

## Completed publication verification

1. The three GitHub release assets were independently downloaded and matched
   their recorded byte counts and hashes.
2. The version and concept DOI resolved without authentication.
3. Every Zenodo record file was inspected; only the generated repository ZIP
   was present.
4. The downloaded Zenodo ZIP matched its recorded size, MD5, and SHA-256.
5. Its extracted repository passed the release verifier and tests.
6. `README.md` and `CITATION.cff` now carry the verified identifiers and
   observed record contents.

## Post-release preservation rules

1. Keep the existing `v1.0.0` tag immutable.
2. Place DOI and record-content corrections in later main-branch metadata
   commits; do not recreate or move the tag.
3. Continue to describe the full sanitized ZIP as a GitHub release asset, not
   a Zenodo file, unless a future Zenodo record is independently observed to
   contain that exact asset.
4. Preserve the version DOI for citations of `v1.0.0` and the concept DOI for
   references intended to follow future software versions.
