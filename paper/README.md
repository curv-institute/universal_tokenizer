# Paper

Build the paper with:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk:

```bash
latexmk -pdf main.tex
```
