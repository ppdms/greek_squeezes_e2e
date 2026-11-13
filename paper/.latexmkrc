# The paper uses fontspec: every pdf build (latexmk -pdf included) runs LuaLaTeX.
$pdflatex = 'lualatex -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 0;
