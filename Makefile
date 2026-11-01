PY ?= python3

.PHONY: all analysis paper clean
all: analysis

analysis:
	cd src && $(PY) 01_preprocess.py && $(PY) 02_eda.py && $(PY) 03_pca.py \
	  && $(PY) 04_cluster.py && $(PY) 05_maps.py && $(PY) 06_mortality.py && $(PY) 07_more_clustering.py

# needs a TeX distribution with XeLaTeX (ctex); or use Overleaf with XeLaTeX
paper:
	cd paper && xelatex main.tex && xelatex main.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.xdv
