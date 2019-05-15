**gmentro.py** is a program to calculate configurational entropy from
a conformational ensemble using Gaussian mixtures. It is an
implementation of the algorithm described in

Gergely Gyimesi, Péter Závodszky, and András Szilágyi:\
**Calculation of configurational entropy differences from
conformational ensembles using Gaussian mixtures**\
_J. Chem. Theory Comput._, 13(1):29-41. (2017)\
DOI:
[10.1021/acs.jctc.6b00837](http://dx.doi.org/10.1021/acs.jctc.6b00837)
[PubMed](https://www.ncbi.nlm.nih.gov/pubmed/27958758)

If you use the program, please cite the above paper.

The program is written in Python (3.x recommended), and needs the
`numpy` module. Much of the code has been adapted from Jakob
Verbeek's [Matlab code for accelerated Gaussian mixture 
learning](http://lear.inrialpes.fr/people/verbeek/software.php).

As a documentation, a manual page is provided in
[gmentro.py.1.md](gmentro.py.1.md), which is also available [as a html
file](gmentro.py.1.html) and a regular [manpage file](gmentro.py.1)
(copy this file into a directory in your MANPATH and you'll be able to
use `man gmentro.py` to display the manual page).

`xvg2dat.py` is a helper program to create a torsion angle trajectory
file from Gromacs coordinate trajectories. See instructions in the
program file.

The Makefile only serves to create the various formats of the man page
(requires the [ronn](https://github.com/rtomayko/ronn) program).
