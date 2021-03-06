gmentro.py(1) -- Calculate entropy by Gaussian mixture fitting
================================================================

## SYNOPSIS

`gmentro.py` [`-h`] [`-d`] [`-q`] [`-c`] [`-w`] [`-J` _J_] [`--order`
`1`|`1.5`|`2`|`full`] [`--center`] [`--centeronly`] [`--stop` `cv`|`aicsd`]
[`--overfit` _OVERFIT_] [`--sdelta` _SDELTA_] [`--emt` _EMT_] [`--cols` _COLS_] [`--slice`
_SLICE_] [`--maxk` _MAXK_] [`--ncand` _NCAND_] [`--unit` `J`|`c`|`e`]
[`--odir` _directory_] [`--version`] _datafilename_

## DESCRIPTION

`gmentro.py` estimates entropy by Gaussian mixture fitting. The data
file named _datafilename_ must contain an _n_ x _d_ matrix with _n_
lines and _d_ columns, representing _n_ samples of _d_ variables.
Comment lines starting with `#` are allowed in the data file.

## OPTIONS

* `-h`:
Show help message and exit

* `-d`:
Turn on debug mode. Generates a lot of extra output.

* `-q`:
Quiet mode (no output to console, only to files)

* `-c`:
Console mode (no output to files, only to console)

* `-w`:
If output files already exist, overwrite them. Default: throw an error
and exit if output files already exist.

* `-J` _J_:
Job number or job id string (for batch jobs or repeated runs; will set `-q`)

* `--order`  `1`|`1.5`|`2`|`full`:
Order of approximation to be used. `1` only calculates 1-D
entropies and sums them up. `1.5` calculates the first-order
approximation and corrects it with the quasiharmonic mutual information.
`2` also calculates pairwise mutual information values and subtracts their 
sum from the first-order entropy to obtain the second-order entropy. 
The sum of mutual information values will also be reported. `full` calculates a
full-dimensional entropy. Default: `full`

* `--center`:
Perform distribution centering before calculating the entropy
(default: no). This option assumes that the data are angles measured
in degrees. Centering is always required for angle data unless the
data in the data file are pre-centered.

* `--centeronly`:
Perform distribution centering on the data and save the centered data;
do not calculate entropy. If the input data file is _data.dat_, the
centered data will saved to _data.centered.dat_.

* `--stop` `cv`|`aicsd`:
Stopping criterion to use. `cv` refers to the cross-validation stopping criterion;
`aicsd` uses a combination of the Akaike Information Criterion and entropy change
upon adding another component (see `--sdelta`). See section STOPPING CRITERION below.
Default: `cv`.

* `--overfit` _OVERFIT_:
Continue the calculation after the stopping criterion is met to obtain _OVERFIT_ extra
Gaussian components. Default: 0. This option is only for testing purposes, and is likely to
result in overfitting.

* `--sdelta` _SDELTA_:
Calculation will stop when the entropy change in the 
last step is below _SDELTA_. Only used with `--stop` `aicsd`. 
See section STOPPING CRITERION below. Default: 0.2.

* `--emt` _EMT_:
Convergence threshold for EM (Expectation Maximization) calculation
(default: 1e-5). This affects how accurately the individual Gaussian
components are fit to the data. A lower threshold will result in
longer calculation.

* `--cols` _COLS_:
Datafile columns to process as list of comma-separated ranges, e.g.
_1-5,8,10-12_. The other columns will be ignored. Default: use all
columns.

* `--slice` _SLICE_:
A string in the format start:stop:step to be used to select lines to
load from the datafile. It follows NumPy syntax, i.e. line numbering 
starts with zero, and the line with index "stop" is not loaded. Example:
10:20:2 means that lines 10, 12, 14, 16 and 18 will be loaded (where line
10 means the 11th line in the file). If any values are omitted, the default
values will be used. Default start is zero, default stop is the end of the file,
and default step is 1. The default slice is "::". This option can be used e.g.
to skip the equilibration stage in a simulation, or to test the entropy
calculation with different sample sizes obtained by changing the time between frames
in a simulation trajectory or limiting the simulation time.

* `--maxk` _MAXK_:
Maximum number of Gaussian components to be used in the mixture
(default: 200). The algorithm will add components until convergence,
but it will stop when _MAXK_ is reached. Set _MAXK_ to 1 to obtain a
quasi-harmonic approximation where only a single Gaussian is fit.

* `--ncand` _NCAND_:
Number of candidates to use in component search (default: 30). The
algorithm will try this many candidate components to find the next
component of the mixture. If it fails to find an appropriate
component, it will report the failure but will repeat the search.
After 100 repetitions (i.e. after trying 100*_NCAND_ candidates),
however, the program will give up and exit. In this case, the _NCAND_
parameter could be increased for an even more thorough search. For
easy cases, _NCAND_ can also be decreased to speed up the run.

* `--unit`  `J`|`c`|`e`:
Entropy unit to use: `J` for J/K/mol, `c` for cal/K/mol, `e` for
natural units (nats). Default: `J`.

* `--odir` _directory_:
Write output files to _directory_ instead of the current directory.

* `--version`:
Print the program's CRC32 checksum and exit. Used to identify program version.


## OUTPUT

The program writes its output to the console, and it also generates 3
output files whose names are formed from the data file name after
cutting off its last 4 characters. For example, if the input file is
named _datafile.dat_ then the following output files will be
generated:

* `datafile.gme.out`:
    Output file. It contains a header section but omits warnings.

* `datafile.gme.log`:
    Log file. Same as output file but includes warnings and debug
    messages. This is identical to what is written on the console.

* `datafile.gme.npz`:
    The parameters of the fitted Gaussian mixture as a numpy npz
    file. This can be loaded by numpy.

If the `-J` _J_ option is specified to provide a job number then the name
of each output file will include the job number after the `.gme.`
part, i.e. `datafile.gme.`_J_`.out`.

The `.out` and `.log` files will contain a header section where the
program parameters are summarized. This is followed by the calculation
results. Lines containing textual information will start with a `#`
character for easier parsing (except in debug mode).

For full-dimensional calculations, the output will be two or three columns,
with the first column being the number of Gaussian components and the
second column the calculated entropy with the given number of
components. This can be used to plot the entropy vs. the number of
components easily, e.g. with `gnuplot`. If the cross-validation stopping
criterion is used (as by default), the second and third columns will contain
the entropy as calculated from the training and testing set, respectively.
Generally, the value in the second column should be used as the estimated entropy.


## STOPPING CRITERION


By default, the program uses the cross-validation stopping criterion. This
means that the input ensemble is randomly divided into two equal parts which will
serve as training and testing sets. The Gaussian mixture fitting is performed on the 
training set and the log-likelihood is evaluated on the testing set upon adding each new 
Gaussian component. If this log-likelihood decreases upon adding a new component, the 
component is discarded and the calculation is stopped.

The cross-validation stopping criterion is very robust, but because the input ensemble is
divided randomly into two parts, the estimated entropy will be different upon running the
program several times. You can calculate a mean and a standard deviation from the results.

As an alternative to the cross-validation stopping criterion, the
program can use a combination of two stopping criteria: the Akaike
Information Criterion (AIC) and an entropy change criterion. This can
be activated by using the `--stop aicsd` option. The program will stop
when either of the two criteria is met.

The AIC is defined by

_AIC_ = 2\*_npar_ - 2\*_LL_

where _LL_ is the log-likelihood of the mixture with the given sample,
and _npar_ is the number of parameters in the mixture:

_npar_ = _k_-1 + _k_\*_d_ + _k_\*_d_\*(_d_+1)/2

which includes the number of weights (_k_-1), the means of the Gaussian
components (_k_\*_d_ parameters), and the covariance matrices
(_k_\*_d_\*(_d_+1)/2 parameters).

The algorithm stops when the AIC increases compared to the previous
step, i.e. if _AIC_(_k_+1) _ _AIC_(_k_) then the _k_-component Gaussian mixture
generated in the previous step will be accepted as the best estimate,
and the associated entropy will be reported.

The entropy change criterion will cause the program to stop when the
calculated entropy differs by less than _SDELTA_ from the one
calculated in the previous step. The value of _SDELTA_ can be set by
the `--sdelta` _SDELTA_ command line option; its default value is 0.1.
Because this is a relatively small value, the AIC will often stop the
program before the entropy change falls below this value.

Note that the algorithm also stops if the log-likelihood does not
increase upon performing partial EM on the newly added Gaussian
component. Therefore, the program can detect, for example, if the
sample was generated from only a single Gaussian distribution.

## WARNINGS

The program can display a number of warnings as it runs.
The most common warnings are:

* `"x out of n likelihoods are too small"`:
This means that the calculated likelihood of some data points is below
the smallest positive number that can be represented by the computer.
This is normal for high-dimensional samples at the beginning of the
calculation. The number of such points should decrease as the
calculation progresses, and finally the warning should disappear. If
the warning remains until the end, even before the final entropy
value, the calculated entropy may not be sufficiently accurate. In
this case, it is recommended that you use fewer variables, a larger
sample, or a different stopping criterion.
	
* `"No appropriate candidates found. Trying more candidates..."`:
The program is having a hard time finding appropriate candidates for
new components, but it keeps trying.
	
* `"Failed to find candidates. Result has not converged."`:
The program tried many times to find candidates but failed. The
displayed result is not valid. You could try to increase the _NCAND_
parameter and rerun. If this keeps failing, try a larger sample, fewer
variables, or a different stopping criterion.
	
* `"No parent can be split, sample too small"`:
This warning appears when so many components have been added that the
number of points in each component has become too small and no more
components can be added (a component must hold at least _d_+1 points)
while the convergence criteria have not been met. In this case, the
result is not acceptable. You must provide a larger sample or reduce
the number of dimensions. This warning is unlikely to occur when the
cross-validation stopping criterion is used.
	
## CONVERGENCE PROBLEMS

Most convergence problems can be avoided by using the cross-validation
stopping criterion (as by default). When using the `--stop aicsd`
option, i.e. using a combination of AIC and entropy change stopping
criteria, convergence problems may appear, especially with
undersampled distributions. The Akaike Information Criterion still
prevents convergence problems in most cases because it stops the
algorithm before the number of components could grow too high.
However, for small samples and/or small number of variables, the AIC
may not be able to detect the optimum model, and the entropy change
criterion will be dominant. In this case, if the _SDELTA_ value
specified with the `--sdelta` option is too small, the algorithm may
fail to converge. For undersampled distributions, the program may keep
adding more and more components, with the entropy decreasing nearly
linearly until the calculation finally reports that no more components
can be added. This typically indicates insufficient sampling; the data
points probably occupy a random intricate shape in the
high-dimensional space. To solve the problem, increase _SDELTA_.
Alternatively, increase sample size or reduce the number of
dimensions. In some cases, the ensemble may actually have an overly
complex shape which cannot be approximated well by a Gaussian mixture
with a reasonable number of components.

## EXAMPLES

* `gmentro.py datafile.dat`:
Calculate full-dimensional entropy on the sample in `datafile.dat`,
using all default parameters.

* `gmentro.py --center datafile.dat`:
Perform distribution centering on the data in `datafile.dat` before
calculating the entropy. The data in the file must be angles measured
in degrees. After the centering, the entropy will be calculated as
usual.

* `gmentro.py --centeronly datafile.dat`:
Perform distribution centering and save the data to
`datafile.centered.dat` then exit. No entropy will be calculated. The
centered data can then be used in further runs of `gmentro.py`.

* `gmentro.py --maxk 1 datafile.dat`:
Calculate quasiharmonic entropy (entropy from a single Gaussian) for
the sample in `datafile.dat`

* `gmentro.py --order 2 datafile.dat`:
Calculate a second-order approximation of the entropy for the sample
in `datafile.dat`. All 1-D entropies and all 2-D entropies and mutual
information values, as well as their sums will be reported.

* `gmentro.py --cols 1-4,8-10 --every 3 datafile.dat`:
Calculate entropy only for the data file columns 1-4 and 8-10
(ignoring all other columns), and only load every 3rd line from the
data file.

* `gmentro.py --aicsd --sdelta 0.1 --emt 1-e6 datafile.dat`:
Calculate full-dimensional entropy for `datafile.dat` using a combination of AIC
and entropy change stopping criterion. Calculation will stop when either the AIC 
criterion is met or the entropy change upon adding a new component is less than 0.1 J/K/mol.
Execution time may be longer and overfitting may occur.

* `gmentro.py --ncand 50 datafile.dat`:
A full-dimensional entropy will be calculated with the number of
candidate Gaussians tried in each step increased to 50 from the
default value of 30. This may increase the chance of convergence for
difficult cases.

## BATCH JOBS AND REPEATED RUNS

It is advisable to run the entropy calculation several times to see
how much the results vary. To start a batch of 6 runs of `gmentro.py`
on the same data, the following Bourne shell script could be used:


    for i in 1 2 3 4 5 6 ; do
      gmentro.py -J $i datafile.dat &
    done


The output files will be named `datafile.gme`._X_._yyy_ where _X_ goes
from 1 to 6 and _yyy_ is one of _out_, _log_, and _npz_.

The `-J` _J_ option is also useful when `gmentro.py` is run several times
on the same data with different parameters. By adding a job number or
job id string, output files from different runs can have different
names, thereby avoiding the overwriting of the output files from the
previous run.

## USAGE FOR ANGLE DATA

Entropy calculation on angle data using Gaussian mixtures requires
that the angle distributions be centered. The distribution centering
can be performed using the `--center` command line option, which
assumes that the angles in the data file are provided in degrees.
Centering can be omitted if the angle data in the data file are
already pre-centered. Pre-centering can be performed using the
`--centeronly` option. Note that entropies calculated on non-centered
angle data will be meaningless.

## AUTHOR

Written by Andras Szilagyi (szilagyi.andras at ttk.mta.hu). Much of the
code was adapted from the Matlab code downloaded from
[http://lear.inrialpes.fr/people/verbeek/software.php](http://lear.inrialpes.fr/people/verbeek/software.php).
The reference for the greedy EM method for Gaussian mixture fitting
is: Verbeek JJ, Vlassis N, Krose B.: Efficient greedy learning of
gaussian mixture models. Neural Comput. 2003 Feb;15(2):469-85.
Please contact the author with bug reports, comments, etc.

## CITATION

Please cite `gmentro.py` as follows:

Gyimesi G, Zavodszky P, Szilagyi A:\
Calculation of configurational entropy differences from conformational ensembles using 
Gaussian mixtures.\
J. Chem. Theory Comput., 13(1):29-41. (2017) DOI: [10.1021/acs.jctc.6b00837](http://dx.doi.org/10.1021/acs.jctc.6b00837)\
PMID: [27958758](https://www.ncbi.nlm.nih.gov/pubmed/27958758)

## WEB SITE

The program can be downloaded from [http://gmentropy.szialab.org](http://gmentropy.szialab.org).
