# 中文报告使用 ctexart，需 XeLaTeX（见 main.tex 顶部 magic comment）。
.PHONY: all clean

all:
	latexmk -xelatex -interaction=nonstopmode main.tex

clean:
	latexmk -C main.tex 2>/dev/null || true
	rm -f main.pdf main.aux main.log main.out main.toc main.fls main.fdb_latexmk main.synctex.gz
