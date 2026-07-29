# Zenodo deposit checklist

Do not invent or prefill a DOI. Zenodo's GitHub integration is configured to
archive the first public `v1.0.0` release and mint its DOI. Record only the
identifier actually generated from that archived release, after it resolves
without authentication and the deposited payload passes read-back
verification.

## Intended payload

- Filename:
  `PLOS_R1_publication_20260729T084516Z_sanitized_data_code_deposit.zip`
- Bytes: `64,452,428`
- SHA-256:
  `8d648613dde0134c828faab507634481a827e22ce16fbfc9e7dd057a03ee0f63`
- Related GitHub release:
  `https://github.com/terry0809000/beyond-macro-f1-pubmed-rct-public/releases/tag/v1.0.0`

The ZIP is deliberately not committed to Git.

The bound ZIP was created before license approval and contains no license
files. Rebuild it for public deposit by adding the exact `LICENSE`,
`LICENSE-CONTENT-DATA-CC-BY-4.0.txt`, and `LICENSES.md` files from this staging
tree. Record the rebuilt ZIP's new byte count and SHA-256 here and in
`README.md`; do not continue to describe the pre-license values above as the
final public payload.

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
boundary prominently in the record description and include `LICENSE`,
`LICENSE-CONTENT-DATA-CC-BY-4.0.txt`, and `LICENSES.md` in the archive. Do not
imply that either license applies to excluded third-party PubMed text,
pretrained models, libraries, fonts, or model weights.

## Draft/reviewer workflow

1. Confirm that this public repository is enabled in Zenodo's GitHub
   integration.
2. Rebuild the exact license-inclusive ZIP and record its byte count and
   SHA-256 before release.
3. Publish the reviewed GitHub `v1.0.0` release so Zenodo archives it.
4. If peer review requires access before final publication, use only a
   reviewer-access mechanism whose link is tested in a signed-out browser.
   Do not commit a token-bearing preview URL to Git.
5. Keep the DOI and manuscript language pending until Zenodo has generated the
   identifier from the archived release.

## Publication verification

1. Confirm that the approved MIT/CC BY 4.0 boundary and third-party exclusions
   are visible in the record, then publish after final author review.
2. Download the deposited ZIP from the public record.
3. Confirm exact byte count and SHA-256.
4. Resolve the generated version and concept DOI(s) in a signed-out browser.
5. Add the verified DOI to `README.md`, `CITATION.cff`, and the manuscript Data
   Availability Statement.
6. Record the immutable Git commit and release tag related to the deposit.
7. Rerun the repository verifier and preserve its JSON output outside Git.
