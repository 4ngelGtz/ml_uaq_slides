TEX=pdflatex
OUTDIR=build
MAIN=main.tex
FIGDIR=slides/figs
FIGSCRIPT=scripts/week01_make_figs.py
WEEK02FIGSCRIPT=scripts/week02_make_beta_priors_fig.py
WEEK03FIGSCRIPT=scripts/week03_make_multivariate_figs.py
WEEK04FIGSCRIPT=scripts/week04_make_mle_erm_figs.py

# Prefer a local venv if present (avoids global Python deps)
PYTHON:=$(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Lista de figuras esperadas (ajusta si agregas/quitas)
FIGS=$(FIGDIR)/supervised_schematic.png \
     $(FIGDIR)/iris_2d.png \
     $(FIGDIR)/overfitting_polyfits.png \
     $(FIGDIR)/error_vs_degree.png \
     $(FIGDIR)/leakage_scaling.png \
     $(FIGDIR)/kfold_schematic.png \
     $(FIGDIR)/pr_curve.png \
     $(FIGDIR)/selection_bias.png

WEEK02FIG=$(FIGDIR)/week02_beta_priors.png

WEEK03FIGS=$(FIGDIR)/week03_cov_corr.png \
     $(FIGDIR)/week03_uncorrelated_dependent.png \
     $(FIGDIR)/week03_simpson.png \
     $(FIGDIR)/week03_mahalanobis.png \
     $(FIGDIR)/week03_covariance_geometry.png \
     $(FIGDIR)/week03_conditioning_2d.png \
     $(FIGDIR)/week03_imputation.png \
     $(FIGDIR)/week03_imputation_iris.png \
     $(FIGDIR)/week03_gmm_2d.png

WEEK04FIGS=$(FIGDIR)/week04_underflow_loglik.png \
     $(FIGDIR)/week04_bernoulli_nll.png \
     $(FIGDIR)/week04_gaussian_nll_mu.png \
     $(FIGDIR)/week04_01_vs_logloss.png

all: $(OUTDIR)/main.pdf

# 1) Regla: generar figuras
$(FIGS): $(FIGSCRIPT)
	mkdir -p $(FIGDIR)
	$(PYTHON) $(FIGSCRIPT)

$(WEEK02FIG): $(WEEK02FIGSCRIPT)
	mkdir -p $(FIGDIR)
	$(PYTHON) $(WEEK02FIGSCRIPT)

# Week 3 figures (MVN, condicionamiento, imputación, GMM)
$(WEEK03FIGS): $(WEEK03FIGSCRIPT)
	mkdir -p $(FIGDIR)
	$(PYTHON) $(WEEK03FIGSCRIPT)

# Week 4 figures (MLE / ERM)
$(WEEK04FIGS): $(WEEK04FIGSCRIPT)
	mkdir -p $(FIGDIR)
	MPLCONFIGDIR="$(CURDIR)/.mplcache" $(PYTHON) $(WEEK04FIGSCRIPT)

# 2) Regla: compilar PDF (depende de figuras y contenido)
$(OUTDIR)/main.pdf: $(MAIN) $(FIGS) $(WEEK02FIG) $(WEEK03FIGS) $(WEEK04FIGS) slides/week01/week01_content.tex slides/week02/week02_probability_univariate.tex slides/week03/week03_probability_multivariate.tex slides/week04/week04_statistics_mle_erm.tex
	mkdir -p $(OUTDIR)
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR) $(MAIN)
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR) $(MAIN)

figs: $(FIGS) $(WEEK02FIG) $(WEEK03FIGS) $(WEEK04FIGS)

clean:
	rm -rf $(OUTDIR)/*
	rm -rf $(FIGDIR)/*.png

.PHONY: all clean figs
