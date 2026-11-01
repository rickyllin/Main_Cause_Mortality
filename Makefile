PY ?= python3

.PHONY: all analysis health_econ paper clean
all: analysis

analysis:
	cd src && $(PY) 01_preprocess.py && $(PY) 02_eda.py && $(PY) 03_pca.py \
	  && $(PY) 04_cluster.py && $(PY) 05_maps.py && $(PY) 06_mortality.py && $(PY) 07_more_clustering.py \
	  && $(PY) 08_bayes_hierarchical.py && $(PY) 09_spatial_clustering.py

health_econ:
	cd src/health_economics && $(PY) 10_panel_prep.py && $(PY) 11_fixed_effects.py && $(PY) 12_did_covid.py \
	  && $(PY) 13_synthetic_control.py && $(PY) 14_dml_did.py && $(PY) 15_parallel_trends_sensitivity.py \
	  && $(PY) 16_spatial_inequality.py

# needs a TeX distribution with XeLaTeX (ctex); or use Overleaf with XeLaTeX
paper:
	cd paper && xelatex main.tex && xelatex main.tex \
	  && xelatex health_economics.tex && xelatex health_economics.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.xdv
