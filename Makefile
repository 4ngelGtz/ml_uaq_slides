TEX=pdflatex
OUTDIR=build
MAIN=main.tex
FIGDIR=slides/figs
FIGSCRIPT=scripts/week01_make_figs.py

# Lista de figuras esperadas (ajusta si agregas/quitas)
FIGS=$(FIGDIR)/supervised_schematic.png \
     $(FIGDIR)/iris_2d.png \
     $(FIGDIR)/overfitting_polyfits.png \
     $(FIGDIR)/error_vs_degree.png \
     $(FIGDIR)/leakage_scaling.png \
     $(FIGDIR)/kfold_schematic.png \
     $(FIGDIR)/pr_curve.png \
     $(FIGDIR)/selection_bias.png

all: $(OUTDIR)/main.pdf

# 1) Regla: generar figuras
$(FIGS): $(FIGSCRIPT)
	mkdir -p $(FIGDIR)
	python $(FIGSCRIPT)

# 2) Regla: compilar PDF (depende de figuras y contenido)
$(OUTDIR)/main.pdf: $(MAIN) $(FIGS) slides/week01/week01_content.tex slides/week02/week02_probability_univariate.tex
	mkdir -p $(OUTDIR)
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR) $(MAIN)
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR) $(MAIN)

figs: $(FIGS)

clean:
	rm -rf $(OUTDIR)/*
	rm -rf $(FIGDIR)/*.png

.PHONY: all clean figs
