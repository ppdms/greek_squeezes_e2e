# Official TROGS-26 evaluation code (verbatim)

Both files are verbatim copies of the organisers' releases. Do not edit them; silence their diagnostic
prints from the caller.

| File | Source | SHA-256 |
|---|---|---|
| `contest_evaluation_v2.py` | TROGS-26 Test Images release (Smith ScholarWorks, `dds_data/19`, 2026-03-27), retrieved from its mirror `huggingface.co/datasets/papadimas/trogs-26-test-images` (`contest_evaluation_v2.py`); ScholarWorks refuses non-browser downloads | `2fafdc5973b748a0fdef84fcb49df51d5a052a546720fbee95792f295338aa85` |
| `contest_evaluation_v1.py` | `contest_evaluation.zip` from the competition page (`science.smith.edu/~nhowe/contest/trogs26.html`), zip SHA-256 `ce4d0dd89df5c2bd35856400789ea16af975e436a1f58e8b1b8c47bde88d622d` | `e3dd63b520aec14850ffdb3c143f5456cf118454bc5095756c05a669017f4d98` |
| `contest_evaluation_v1_README.md` | `README.md` from the same zip | `eabe30be725f5054f9b52dbf9cbc005677ab9c1f5e9195426a101c2948273dc3` |

Author: Nicholas Howe. Licence: CC-BY 4.0.

Use: v2 is the official scorer (Core = `cer`, confident letters only). It also supplies the row grouping the decoder
applies to detected boxes (`Box`, `orderBoxes`, `getRowBoxes`). v1's `getRowTranscript` extracts the TRAIN
transcripts the character language model is built from.
