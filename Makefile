# Makefile to generate man page in roff, html, and markdown

all: gmentro.py.1 gmentro.py.1.html gmentro.py.1.md

gmentro.py.1: gmentro.py.man.ronn
	ronn -r < $< > $@

gmentro.py.1.html: gmentro.py.man.ronn
	ronn -5 < $< > $@

gmentro.py.1.md: gmentro.py.man.ronn
	tr '<>' __ < $< > $@
	
