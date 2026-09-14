# Paper offline kit (Mac / Overleaf)

Everything needed to **write the Block‑1 prediction paper** without the Jetson.

| Path | Contents |
|------|----------|
| [README_PAPER_KIT.txt](README_PAPER_KIT.txt) | Start here — checklist + numbers |
| [CLAIM_AND_LIMITS.txt](CLAIM_AND_LIMITS.txt) | What you may / may not claim |
| [METHOD_CHEATSHEET.txt](METHOD_CHEATSHEET.txt) | Equations & symbols |
| [LATEX_FIGURE_SNIPPETS.tex](LATEX_FIGURE_SNIPPETS.tex) | Drop-in figure LaTeX |
| [figures/](figures/) | PNGs for the paper |
| [data/](data/) | ADE table + per-frame CSVs |
| [paper_snippets/](paper_snippets/) | `root.tex`, `refs.bib` |

## Copy to Mac

```bash
scp -r user@<jetson>:~/ros2_ws/src/zed-detection/zed-prediction/docs ~/Desktop/zed-paper-docs
```

Or commit/push this folder and pull on the Mac.
