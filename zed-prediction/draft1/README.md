# ICRA LaTeX draft

Official RAS / PaperCept template, downloaded 12 Sep 2026.

**Do not use Underleaf, Overleaf clones, or random GitHub “ICRA templates” as the source of truth.** ICRA 2027 points authors to PaperCept, and PaperCept ships `ieeeconf.cls` (IEEEtran-based).

## Official source

- Conference CFP: [ICRA 2027 Call for Technical Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
- Template page: [PaperCept LaTeX support](https://ras.papercept.net/conferences/support/tex.php)
- Files PaperCept tells you to download:
  - [`ieeeconf.zip`](https://ras.papercept.net/conferences/support/files/ieeeconf.zip) — `ieeeconf.cls`, sample `root.tex`, sample `root.pdf`
  - [`IEEEtranBST.zip`](https://ras.papercept.net/conferences/support/files/IEEEtranBST.zip) — `IEEEtran.bst`
  - [`IEEEtran_HOWTO.pdf`](https://ras.papercept.net/conferences/support/files/IEEEtran_HOWTO.pdf)

Verified contents of the zip we pulled:

| File | Role |
|------|------|
| `official/ieeeconf/ieeeconf.cls` | Class file (do not edit) |
| `official/ieeeconf/root.tex` | Official sample / formatting examples |
| `official/ieeeconf/root.pdf` | Compiled official sample |
| `official/bst/IEEEtran.bst` | IEEE numbered bibliography style |

## ICRA 2027 rules that affect this file

- **8 pages total**, including figures, tables, acknowledgments, and references. Over 8 pages is desk-rejected.
- **US Letter**, 10 pt, two-column conference layout.
- **Double-anonymous review.** No author names, affiliations, grants, GitHub user names, or “our lab” identifiers in the PDF. Enter authors only in PaperPlaza.
- Required class line (Letter, not A4):

```latex
\documentclass[letterpaper, 10 pt, conference]{ieeeconf}
\IEEEoverridecommandlockouts
\overrideIEEEmargins
```

- Optional video: mpeg/mp4/mpg, ≤20 MB, ≤180 s. Video windows are separate from the PDF deadline.
- Regular paper deadline: **15 Sep 2026, 23:59 PST.** No extension announced.

## Where to write

Write in `paper/root.tex`. That folder already has copies of `ieeeconf.cls` and `IEEEtran.bst` so it compiles on its own.

```bash
cd zed-prediction/draft1/paper
pdflatex root.tex
bibtex root
pdflatex root.tex
pdflatex root.tex
```

Keep `official/` untouched. Use `paper/official_sample_root.tex` if you need a formatting example (headings, figures, tables).

Figures for results live in `../results/paper_1s/`. The draft already points `\graphicspath` there.
