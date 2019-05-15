#!/usr/bin/env python3

"""
creates one big datafile from gromacs xvg files containing torsion angle histories
Written by Andras Szilagyi, 2015

Run the following Gromacs command in your directory containing Gromacs simulation results:

gmx chi -s confout.gro -f traj.trr -all -psi -phi -maxchi 6

where traj.trr is your trajectory file and confout.gro is the final
coordinate file from the simulation. This will result in a large
number of files, each containing the trajectory for a single torsion
angle.

Then change to the parent directory and use xvg2dat.py to merge all
the torsion data into a single file. 

The program can merge data from multiple directories (specify all directory
names on the command line).
"""

import sys
import os
import argparse
from operator import itemgetter

def residsort(rname):
    """key for sorting residue filenames"""
    if rname[:3] == 'chi':
        rnum = int(rname[7:-4])
        angx = int(rname[3])+2
    else:
        rnum = int(rname[6:-4])
        if rname[:3] == 'phi':
            angx = 1
        else:
            angx = 2
    return rnum,angx

parser = argparse.ArgumentParser(description='Create one big datafile from gromacs xvg '+
           'files containing torsion angle histories')
parser.add_argument('-t', '--angtype',choices=['all','phipsi','chi'], default='all',
                    help='Type of angles to extract, default: all')
parser.add_argument('-o', default='torsions', help='Output file name base (default: "torsions")',
                    metavar='basename')
parser.add_argument('--order', choices=['time', 'dir'], default='time', 
                    help='Sort order: by time (default) or order of directories')
parser.add_argument('--skip', default=0, type=int, help='Skip first N frames', metavar='N')
parser.add_argument('--term', action='store_true', 
                    help='Include phi for N- and psi for C-terminus (default: omit)')
parser.add_argument('directory', nargs='+', help='Directory/ies containing the xvg files')

a = parser.parse_args()

dirs = list(map(os.path.normpath, a.directory))

angtypdic = {'all':['phi','psi','chi'],'phipsi':['phi','psi'],'chi':['chi']}
angtypes = angtypdic[a.angtype]

# extract and order residue list

dili = os.listdir(dirs[0])
fnames = [fn for fn in dili if fn.endswith('.xvg') and fn[:3] in angtypes]
fnames.sort(key=residsort) # sort by residue number and angle index

# if --term is not given, omit N-terminal phi and C-terminal psi
if not a.term:
    fnames = fnames[1:] # remove first psi angle
    angs = list([s[:3] for s in fnames])
    x = len(angs)-1-angs[::-1].index('psi') # index of last psi angle
    del fnames[x]
    
# set output file names
skipstr = '.skip'+str(a.skip) if a.skip > 0 else ''
termstr = '.nc.' if a.term else ''
cfile = a.o+'.'+a.angtype+skipstr+termstr+'.columnresnum'
ofile = a.o+'.'+a.angtype+skipstr+termstr+'.dat'

# write columnresnum file
fw = open(cfile,'w')
for j,fn in enumerate(fnames):
    fw.write('%d %s\n' % (j+1,fn[:-4]))
fw.close()

# extract angles and times
tangles = []
for j in range(len(fnames)+1):
    tangles.append([])

for direc in dirs:
    dili = os.listdir(direc)

    # make sure file names in this dir match fnames
    fnames2 =[fname for fname in dili if fname.endswith('.xvg') and 
            fname[:3] in angtypes]
    fnames2.sort(key=residsort)
    if not a.term:
        fnames2 = fnames2[1:] # remove first psi angle
        angs = list([s[:3] for s in fnames2])
        x = len(angs)-1-angs[::-1].index('psi') # index of last psi angle
        del fnames2[x]
    if fnames2 != fnames:
        sys.stderr.write('Filename list in %s differs from that in %s, aborting\n' % 
                         (direc,dirs[0]))
        sys.exit()

    # extract times into first list in tangles
    nl = 0
    for line in open(os.path.join(direc, fnames[0])):
        if line[0] in '#@':
            continue
        nl += 1
        if nl > a.skip:
            tangles[0].append(float(line.split()[0]))

    # extract angles into tangles
    for (j, fname) in enumerate(fnames):
        nl = 0
        for line in open(os.path.join(direc, fname)):
            if line[0] in '#@':
                continue
            nl += 1
            if nl > a.skip:
                if j+1==0:
                    print('j+1=0')
                    sys.exit()
                tangles[j+1].append(line.split()[1])
    
# transpose, sort by time if requested
data = list(zip(*tangles))
if a.order == 'time':
    data.sort(key=itemgetter(0))

# write output data file

fw = open(ofile,'w')

header = '# Torsion trajectory from directories "%s"\n' % ' '.join(dirs)
header += '# See column definitions in file "%s"\n' % cfile
header += '# Frames sorted by %s\n' % a.order

fw.write(header)

for row in data:
    fw.write(' '.join(row[1:])+'\n')
fw.close()
