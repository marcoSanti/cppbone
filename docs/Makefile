# Builds the cppbone API documentation with doxygen.
#
#   make          HTML and LaTeX sources (build/html, build/latex)
#   make html     HTML only
#   make latex    LaTeX sources only
#   make pdf      LaTeX sources, then build/latex/refman.pdf (needs pdflatex and makeindex)
#   make clean    remove build/

DOXYGEN ?= doxygen
BUILD   := build

.PHONY: all html latex pdf clean

all:
	$(DOXYGEN) Doxyfile

# Same configuration with one output switched off, passed to doxygen on stdin.
html:
	( cat Doxyfile; echo "GENERATE_LATEX = NO" ) | $(DOXYGEN) -

latex:
	( cat Doxyfile; echo "GENERATE_HTML = NO" ) | $(DOXYGEN) -

pdf: latex
	$(MAKE) -C $(BUILD)/latex

clean:
	rm -rf $(BUILD)
