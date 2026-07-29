# License scope and attribution

Copyright © 2026 Tianyi Yu.

This release uses two licenses because it contains both software and research
content.

## Author-generated code — MIT

Author-generated software is licensed under the MIT License in `LICENSE`.
This includes:

- Python source under `src/`, `scripts/`, and `tests/`;
- `analysis/run_smoke_analysis.py`;
- author-generated code cells in
  `analysis/certified_source_notebook.ipynb`;
- the continuous-integration workflow; and
- software configuration needed to install or test the code.

## Author-generated content and derived outputs — CC BY 4.0

Author-generated documentation, figures, figure source data, derived result
tables, and non-code notebook content are licensed under the Creative Commons
Attribution 4.0 International license. The complete legal text is in
`LICENSE-CONTENT-DATA-CC-BY-4.0.txt` and the canonical license URI is:

<https://creativecommons.org/licenses/by/4.0/>

This includes:

- `README.md`, `CHANGELOG.md`, `ZENODO_DEPOSIT_CHECKLIST.md`, and release
  documentation;
- `figures/`;
- `artifacts/summary/`;
- author-generated non-code content in the certified notebook; and
- author-generated provenance and audit records, to the extent Tianyi Yu owns
  the applicable rights.

Recommended attribution:

> Tianyi Yu (2026), *Beyond Macro-F1: PubMed RCT
> sentence-classification reproducibility package*, version 1.0.0,
> <https://github.com/terry0809000/beyond-macro-f1-pubmed-rct-public>.

When sharing modified CC BY material, identify the creator, link the license,
link the source where reasonably practicable, and indicate that changes were
made.

## Exclusions and third-party rights

These license grants apply only to material for which Tianyi Yu holds the
necessary rights. They do not relicense:

- PubMed RCT raw or normalized sentence text, which is not distributed here
  and must be obtained from the pinned upstream source under its own terms;
- third-party pretrained models, tokenizers, libraries, fonts, or model
  weights;
- third-party names, logos, or trademarks; or
- any material listed as not redistributed in `WITHHELD_ARTIFACTS.csv`.

The exact certified notebook is a mixed file: its author-generated code cells
are MIT-licensed, while author-generated explanatory markdown and embedded
non-code content are CC BY 4.0-licensed. No raw benchmark text or retired
manual-review dataset is embedded in that notebook.
