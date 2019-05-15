#!/usr/bin/env python3

"""
Entropy estimation using Gaussian mixtures.
Please cite: 
Gyimesi G, Zavodszky P, Szilagyi A:
Calculation of configurational entropy differences from conformational ensembles using 
Gaussian mixtures.
J. Chem. Theory Comput., (Just Accepted Manuscript)
DOI: 10.1021/acs.jctc.6b00837
http://gmentropy.szialab.org/

Gaussian mixture fitting adapted from the matlab code downloaded 
from http://lear.inrialpes.fr/people/verbeek/software.php
journal ref:
Verbeek JJ1, Vlassis N, Krose B.
Efficient greedy learning of gaussian mixture models.
Neural Comput. 2003 Feb;15(2):469-85.
PMID:     12590816 / CT 1817

Code adapted and modified by Andras Szilagyi

"""

import logging
import os.path
import sys
import time
from numpy import *
import numpy.linalg as la
import numpy.random as nra
from zlib import crc32

# important constants
REALMAX = finfo(float64).max
REALMIN = finfo(float64).tiny
EPS = finfo(float64).eps


class tee:
    """write print output to multiple files simultaneously"""
    
    def __init__(self,*fdesc):
        """specify a list of file descriptors to write output to"""
        fdesc = list(set(fdesc)) # make it unique
        self.fdesc = fdesc
    
    def tprint(self,str):
        """prints str to stdout and all files in self.fdesc"""
        for fw in self.fdesc:
            fw.write(str+'\n')
        
            
def tompipi(A):
    """Shift values into the [-180,180) interval. Note: modifies array in place."""
    A[A >= 180] -= 360
    A[A < -180] += 360


def center_angle_distribution(A):
    """
    Shift all variables in a sample of angles to center the distributions.
    A: array of angles, columns are variables, rows are samples.
    Does not modify the input array, returns a new array.
    """
    
    # array for results
    B = A.copy()
    
    # are all values integers?
    intang = all(A == floor(A))
    
    # loop over degrees of freedom
    for k in range(A.shape[1]):
        # make a histogram
        # determine histogram bins depending on whether we have integer or non-integer angles
        if intang:
            rng = linspace(-180.5,179.5, num=361)
        else:
            rng = linspace(-180.0,180.0, num=361)
        nn,ee = histogram(A[:,k], rng)
        H = hstack(((rng[:-1]+0.5).reshape((-1,1)), nn.reshape((-1,1))))
        hl = H.shape[0]
        
        # find the minima
        rr = (H[:,1] == min(H[:,1])).nonzero()[0]
        # duplicate the interval so that the beginning and the end get merged
        rr = hstack((rr, rr+hl))
        # find the longest such interval
        maxl = 0
        maxp = 0
        l = 1
        p = rr[0]
        for i in range(1, len(rr)):
            if rr[i] == rr[i-1]+1:
                l = l+1
            else:
                if l > maxl:
                    maxp = p
                    maxl = l
                l = 1
                p = rr[i]
    
        if maxp > hl:
            maxp -= hl
    
        nl = int(round(maxp+maxl/2.0))
        if nl >= hl:
            nl -= hl
        # the new zero position
        nl = H[nl,0]+180
        if nl >= 180:
            nl -= 360
        if nl < -180:
            nl += 360
    
        # shift the values
        B[:,k] = A[:,k]-nl
        #print "shifting dimension %d by %d" % (k,nl)
        B[B[:,k] >= 180, k] -= 360
        B[B[:,k] < -180, k] += 360
        
    return B


def klist(kstr):
    """process string of comma-separated ranges"""
    cs = kstr.split(',')
    ks = []
    for c in cs:
        if '-' not in c:
            ks.append(int(c))
        else:
            [k1,k2] = map(int, c.split('-'))
            ks = ks+list(range(k1, k2+1))
    return ks


def em_step_partial(X,W,M,R,P,n_all):
    #function [W,M,R] = em_step_partial(X,W,M,R,P,n_all,plo)
    
    n,d = X.shape
    n1=ones((n,1))
    d1=ones((1,d))

    Psum = sum(P,axis=0)
    
    for j in range(len(W)):
        if Psum[j] > REALMIN:
            W[j] = Psum[j] / n_all
            M[j,:] = dot(P[:,j].T,X) / Psum[j]
            Mj = X-n1.dot(M[j,:].reshape((1,d)))
            Sj = dot((Mj*(reshape(P[:,j],(n,1)).dot(d1))).T,Mj) / Psum[j]
            # check for singularities
            U,L,V = la.svd(Sj)  # get smallest eigenvalue
            if L[d-1] > REALMIN:
                try:
                    Rj = la.cholesky(Sj).T
                except la.LinAlgError:
                    pass  # matrix not positive definite
                else:
                    R[j,:] = Rj.flatten(1)
    return (W,M,R)


def rand_split(P,X,M,R,sigma,F,W,nr_of_cand):
    # function [Mus, Covs, Ws]=rand_split(P,X,M,R,sigma,F,W,nr_of_cand)
    
    k = R.shape[0]
    n,d = X.shape
    # threshold in relative loglikelihood improvement for convergence in local partial EM
    epsilon = 1e-2  
    #C=cov(X,rowvar=0)
    #if len(C.shape) == 0: # 0-dim
    #    mineigx = C+0
    #else:
    #    mineigx = min(la.eigvals(cov(X,rowvar=0))) # added by A. Szilagyi
    
    I = argmax(P,axis=1)
    
    Mus = []
    Covs = []
    K = []
    Ws = []
    KL = []
    
    for i in range(k):
        
        XI = (I == i).nonzero()[0]
        Xloc = X[XI,:]
        start = len(Mus)
        j = 0

        Ws = list(Ws)
        Mus = list(Mus)
        Covs = list(Covs)
        if len(XI) > 2*d:  # generate candidates for this parent
            while j < nr_of_cand:  # number of candidates per parent component
                r  = nra.permutation(len(XI))
                r  = r[:2]
                if d == 1:
                    cl = hstack((Xloc-Xloc[r[0]],Xloc-Xloc[r[1]])) 
                    cl = argmin(cl**2,axis=1)
                else:
                    cl = sqdist(Xloc.T, Xloc[r,:].T )
                    cl = argmin(cl,axis=1)
                for guy in range(2):
                    data = Xloc[cl == guy, :] 
                    if data.shape[0] > d:
                        Rloc = cov(data,rowvar=0) + eye(d)*EPS
                        try:
                            Rloc = la.cholesky(Rloc).T
                        except la.LinAlgError: # matrix not positive definite
                            continue
                        j = j+1
                        Mus.append(mean(data,axis=0))
                        Covs.append(Rloc.flatten(1))
                        Ws.append(W[i]/2)
                        Knew = zeros((n,1))
                        Knew[XI] = em_gauss(Xloc,array([Mus[-1]]),array([Covs[-1]]))
                        K.append(Knew)
    
        last = len(Mus)
        Ws = array(Ws)
        Mus = array(Mus)
        Covs = array(Covs)
        
        if last > start: # if candidates were added, do local partial EM
            alpha = Ws[start:last]
            K2 = hstack(K[start:last])[XI,:]        # K(XI,start+1:last)
            Mnew = vstack(Mus[start:last])          # Mus(start+1:last,:)
            Rnew = vstack(Covs[start:last])         # Covs(start+1:last,:)
            FF = F[XI].dot(ones((1,last-start)))
            PP = FF*(ones((len(XI),1)).dot((1-alpha).T))+K2*(ones((len(XI),1)).dot(alpha.T))
            Pnew = (K2*(ones((len(XI),1)).dot(alpha.T)))/PP
            OI = ones((n,1))
            OI[XI] = 0
            OI = (OI == 1).nonzero()[0]
            lpo = sum(log(F[OI]),axis=0)
            ll = sum(log(PP),axis=0) + len(OI)*log(1-alpha.flatten())+lpo
            ll = ll/n
            done = False
            iter_ = 1
            while not done:
                (alpha,Mnew,Rnew) = em_step_partial(Xloc,alpha,Mnew,Rnew,Pnew,n) 
                K2 = em_gauss(Xloc,Mnew,Rnew)
                Fnew = FF*(ones((len(XI),1)).dot((1-alpha).T))+K2*(ones((len(XI),1)).dot(alpha.T))
                old_ll = ll
                ll = sum(log(Fnew),axis=0)+len(OI)*log(1-alpha.flatten())+lpo 
                ll = ll/n
                done = max(abs(ll/old_ll -1)) < epsilon
                if iter_>20:
                    done = True
                iter_=iter_+1
                Pnew = (K2*(ones((len(XI),1)).dot(alpha.T)))/Fnew
            Pnew[Pnew < EPS] = EPS
            Pnew[Pnew == 1] = 1-EPS
            Ws[start:last] = alpha
            Mus[start:last,:] = Mnew
            Covs[start:last,:] = Rnew
            KL += (n*log(1-alpha.T)-sum(log(1-Pnew),axis=0)).flatten().tolist()
    
    KL = array(KL)
    I = []
    logging.debug('%d candidates' % Ws.shape[0])
    
    for i in range(Ws.shape[0]): # remove some candidates that are unwanted
        S = reshape(Covs[i,:],(d,d))
        S = S.T*S
        S = min(la.eigvals(S))
        #if (S<sigma/400 || Ws(i)<2*d/n || Ws(i)>.99) 
        # criterion modified by Andras Szilagyi
        #if (S<sigma/10000)
        #  fprintf("%d S<sigma/10000 %f %f\n",i,S,sigma);
        #elseif (Ws(i)<2*d/n)
        #  fprintf("%d Ws(i)<2*d/n",i);
        #elseif (Ws(i)>.99)
        #  fprintf("%d Ws(i)>.99",i);
        #endif
        #if (S<mineigx/20 || Ws(i)<2*d/n || Ws(i)>.99)
        # criterion modified by A Szilagyi
        if S < sigma/10000 or Ws[i] > .99 or Ws[i] < 0.5*d/n: # < 2.0*d/n
            if S<sigma/10000:
                logging.debug('%d S<sigma/10000 %f %f' % (i,S,sigma/10000))
            if Ws[i]<0.5*d/n: # 2.0*d/n:
                logging.debug('%d Ws[i]<2.0*d/n %f %f %f' % (i,Ws[i],2.0*d/n,2.0*d/n/Ws[i]))
            if Ws[i]>.99:
                logging.debug('Ws[i]>.99')
            I.append(i)
    
    returnstatus = 'ok'
    if Ws.shape[0] == 0:
        logging.warning('No parent can be split, sample too small. Result did not converge.')
        returnstatus = 'split_impossible'
    elif Ws.shape[0] == len(I):
        returnstatus = 'no_candidate_met_criteria'
        
    Ws = delete(Ws,I)
    KL = delete(KL,I)
    Mus = delete(Mus,I,axis=0)
    Covs = delete(Covs,I,axis=0)
    logging.debug('deleting %d candidates, %d remains' % (len(I),Ws.size))
    
    if Ws.size == 0:
        Ws = 0
    else:
        sup = argmax(KL)
        Mus = Mus[sup,:].reshape((1,Mus.shape[1]))
        Covs = Covs[sup,:].reshape((1,Covs.shape[1]))
        Ws = Ws[sup]
    return (Mus,Covs,array([Ws]),returnstatus)


def sqdist(a,b):
    #function d = sqdist(a,b)
    # sqdist - computes pairwise squared Euclidean distances between points
    # vectors are in columns of a and b
    # original version by Roland Bunschoten, 1999
    
    if len(a.shape) == 1:  # we have scalars
        d = tile(a.reshape((-1,1)),(1,len(b))) - tile(b,(len(a),1))
        d = d**2
    else:
        aa = sum(a*a,axis=0)
        bb = sum(b*b,axis=0)
        ab = dot(a.T,b) 
        d = abs(tile(aa.reshape(-1,1),(1,len(bb))) + tile(bb,(len(aa),1)) - 2*ab)
    return d


def kmeans(X,kmax):
    #function [Er,M,nb] = kmeans(X,T,kmax,dyn,bs, killing, pl)
    # kmeans - clustering with k-means (or Generalized Lloyd or LBG) algorithm
    #
    # [Er,M,nb] = kmeans(X,T,kmax,dyn,dnb,killing,pl)
    #
    # X    - (n x d) d-dimensional input data
    # T    - (? x d) d-dimensional test data
    # kmax - (maximal) number of means
    # dyn  - 0: standard k-means, unif. random subset of data init. 
    #        1: fast global k-means
    #        2: non-greedy, just use kdtree to initiallize the means
    #        3: fast global k-means, use kdtree for potential insertion locations  
    #        4: global k-means algorithm
    # dnb  - desired number of buckets on the kd-tree  
    # pl   - plot the fitting process
    #
    # returns
    # Er - sum of squared distances to nearest mean (second column for test data)
    # M  - (k x d) matrix of cluster centers; k is computed dynamically
    #
    # Nikos Vlassis & Sjaak Verbeek, 2001, http://www.science.uva.nl/~jverbeek

    Er = [] 
    #TEr = []              # error monitorring
    
    n,d = X.shape
    
    THRESHOLD = 1e-4   # relative change in error that is regarded as convergence
    #nb = 0  
    
    # initialize 
    k = kmax
    tmp = nra.permutation(n)
    M = X[tmp[:k],:]
    Wold = REALMAX
    
    while k <= kmax:
        kill = []
        
        # squared Euclidean distances to means; Dist (k x n)
        Dist = sqdist(M.T,X.T)
        
        # Voronoi partitioning
        Dwin = amin(Dist.T,axis=1)
        Iwin = argmin(Dist.T,axis=1)
        
        # error measures and mean updates
        Wnew = sum(Dwin,axis=0)
         
        # update VQ's
        for i in range(M.shape[0]):
            I = (Iwin == i)
            if I.shape[0]>d:
                M[i,:] = mean(X[I,:],axis=0)
            elif killing == 1:
                kill.append(i)
        
        if 1-Wnew/Wold < THRESHOLD:
            k = kmax+1

        Wold = Wnew
    
    Er.append(Wnew)
    return (Er,M)


def em_gauss(X,M,R):
    #function L = em_gauss(X,M,R)
    #em_gauss - compute likelihoods for all points and all components
    #
    #L = em_gauss(X,M,R)
    #  X - (n x d) matrix of input data
    #  M - (k x d) matrix of components means
    #  R - (k x d^2) matrix of Cholesky submatrices of components covariances
    #      in vector reshaped format. To get the covariance of component k:
    #      Rk = reshape(R(k,:),d,d); S = Rk'*Rk;
    #returns 
    #  L - (n x k) likelihoods of points x_n belonging to component k
    
    # Nikos Vlassis, 2000

    n,d = X.shape

    k = M.shape[0]

    L = zeros((n,k)) 
    for j in range(k):

        # Cholesky triangular matrix of component's covariance matrix
        if k == 1:
            Rj = reshape(R,(d,d),order='F').copy()
        else:
            Rj = reshape(R[j,:],(d,d),order='F').copy()        
        
        # We need to compute the Mahalanobis distances between all inputs
        # and the mean of component j; using the Cholesky form of covariances
        # this becomes the Euclidean norm of some new vectors 
        New = dot(X - tile(M[j,:],(n,1)), la.inv(Rj))
        Mah = sum(New**2,axis=1)
    
        detrj = la.det(Rj)
        #logging.debug('determinant for component %d: %g' % (j, detrj))
        L[:,j] = (2*pi)**(-d/2.0) / detrj * exp(-0.5*Mah)

    return L


def em_step(X,W,M,R,P):
    # function [W,M,R] = em_step(X,W,M,R,P,plo)
    #em_step - EM learning step for multivariate Gaussian mixtures
    #
    #[W,M,R] = em_step(X,W,M,R,P,plo)
    #  X - (n x d) matrix of input data
    #  W - (k x 1) vector of mixing weights
    #  M - (k x d) matrix of components means
    #  R - (k x d^2) matrix of Cholesky submatrices of components covariances
    #      in vector reshaped format. To get the covariance of component k:
    #      Rk = reshape(R(k,:),d,d); S = Rk'*Rk;
    #  P - (n x k) posterior probabilities of all components (from previous EM step)
    #  plo - if 1 then plot ellipses for 2-d data
    #returns
    #  W - (k x 1) matrix of components priors
    #  M - (k x d) matrix of components means
    #  R - (k x d^2) matrix of Cholesky submatrices of components covariances
    
    # Nikos Vlassis, 2000

    n,d = X.shape
    
    Psum = sum(P,axis=0)

    for j in range(len(W)):
        if Psum[j] > EPS:
            # update mixing weight
            W[j] = Psum[j] / n
            # update mean
            M[j,:] = dot(P[:,j].T,X) / Psum[j]
          
            # update covariance matrix
            Mj = tile(M[j,:],(n,1))
            Sj = ((dot(((X - Mj)*tile(P[:,j].reshape((-1,1)),(1,d))).T, (X - Mj))) / 
                  tile(Psum[j],(d,d)))
        
            # check for singularities
            U,l,V = la.svd(Sj)
            if min(l) > EPS and max(l)/min(l) < 1e4:
                try:
                    Rj = la.cholesky(Sj).T
                except la.LinAlgError: # matrix not positive definite
                    pass
                else:
                    R[j,:] = Rj.flatten(1)
    return W,M,R


def em_init_km(X,k,dyn):
    #function [W,M,R,P,sigma] = em_init_km(X,k,dyn)
    #em_init_km - initialization of EM for Gaussian mixtures 
    #
    #[W,M,R,P,sigma] = em_init_km(X,k,dyn)
    #  X - (n x d) matrix of input data 
    #  k - initial number of Gaussian components
    #  dyn - if 1 then perform dynamic component allocation else normal EM 
    #returns
    #  W - (k x 1) vector of mixing weights
    #  M - (k x d) matrix of components means
    #  R - (k x d^2) matrix of Cholesky submatrices of components covariances
    #  P - (n x k) the posteriors to be used in EM step after initialization
    #  of priors, means, and components covariance matrices

    # Nikos Vlassis & Sjaak Verbeek 2002
    
    n,d = X.shape

    tmp,M = kmeans(X,k)
    DI = sqdist(M.T,X.T)
    #D = amin(DI,axis=0)
    I = argmin(DI,axis=0)

    # mixing weights
    W = zeros((k,1))
    for i in range(k):
        W[i,0] = count_nonzero(I == i)/n

    # covariance matrices 
    R = zeros((k,d**2))
    if k > 1:
        for j in range(k):
            J = (I == j)
            if count_nonzero(J) > 2*d:
                Sj = cov(X[J,:],rowvar=0)
            else:
                Sj = cov(X,rowvar=0)
            Rj = la.cholesky(Sj).T
            R[j,:] = Rj.flatten(1)
    else:
        S = cov(X,rowvar=0)
        if len(S.shape) == 0: # 0-dimensional array
            R = sqrt(S)
        else:
            R = la.cholesky(S).T
        R = array([R.flatten(1)])


    # compute likelihoods L (n x k)
    L = em_gauss(X,M,R)

    # compute mixture likelihoods F (n x 1)
    F = dot(L, W)
    F[F < REALMIN] = REALMIN

    # compute posteriors P (n x k)
    P = L * tile(W.T,(n,1)) / tile(F,(1,k))

    sigma = 0.5 * (4.0/(d+2)/n)**(1.0/(d+4)) * sqrt(la.norm(cov(X,rowvar=0)))
    return (W,M,R,P,sigma)


def em(X, Y, kmax, nr_of_cand, stop, Sdelta, emthresh, ufac, overfit, printout):
    #function [W,M,R,Tlogl,Sk] = myem(X,T,kmax,nr_of_cand,plo,dia,Sdelta,emthresh)
    # em - EM algorithm for adaptive multivariate Gaussian mixtures
    #
    #[W,M,R,Tlogl] = em(X,T,kmax,nr_of_cand,plo,dia)
    #  X     - (n x d) d-dimensional zero-mean unit-variance data
    #  Y     - (m x d) test data for cross-validation (optional, set [] if none)
    #  kmax  - maximum number of components allowed
    #  nr_of_cand - number of candidates per component, zero gives non-greedy EM
    #  plo   - if 1 then plot ellipses for 2-d data
    #  dia   - if 1 then print diagnostics
    #  Sdelta - stop if change in entropy is < Sdelta J/K/mol (added by A. Szilagyi)
    #  emthresh - threshold for EM optimization
    #  overfit - nr of components to calculate after stopping crit is met
    #returns
    #  W - (k x 1) vector of mixing weights
    #  M - (k x d) matrix of components means
    #  R - (k x d^2) matrix of Cholesky submatrices of components covariances
    #      in vector reshaped format. To get the covariance of component k:
    #      Rk = reshape(R(k,:),d,d); S = Rk'*Rk;
    #  Tlogl -  average log-likelihood of test data
    #  Sk - entropy history (added by A. Szilagyi)
    #
    # Nikos Vlassis & Sjaak Verbeek, oct 2002
    # see greedy-EM paper at http://www.science.uva.nl/~vlassis/publications
    
    n,d = X.shape

    converged = False
    
    #n1 = ones((n,1))
    #d1 = ones((1,d))
    
    Sk = zeros((kmax,1))
    
    THRESHOLD = emthresh
    
    if nr_of_cand > 0:
        k = 1
        logging.debug('Greedy EM initialization')
    else:
        k = kmax
        logging.debug('Non-greedy EM initialization')
    
    (W,M,R,P,sigma) = em_init_km(X,k,0)
    oldW,oldM,oldR = W,M,R 

    # do autoscaling of the data to prevent numerical overflows/underflows
    llcorrection = 0.0
    if k == 1:
        Rj = reshape(R,(d,d),order='F').copy()
    else:
        Rj = reshape(R[0,:],(d,d),order='F').copy()        
    detrj = la.det(Rj) # determinant of covariance matrix
    if detrj > 1e20:
        scalingf = (1.0e20/detrj)**(1.0/d)
        X = scalingf*X
        Y = scalingf*Y
        llcorrection = d*log(scalingf)
        logging.debug('# autoscaling data by a scaling factor: %g' % scalingf)
        (W,M,R,P,sigma) = em_init_km(X,k,0)
        oldW,oldM,oldR = W,M,R 
    elif detrj < 1.0:
        scalingf = (1.0/detrj)**(1.0/d)
        X = scalingf*X
        Y = scalingf*Y
        llcorrection = d*log(scalingf)
        logging.debug('# autoscaling data by a scaling factor: %g' % scalingf)
        (W,M,R,P,sigma) = em_init_km(X,k,0)
        oldW,oldM,oldR = W,M,R 
        
    #
    
    sigma = sigma**2
    
    oldlogl = -REALMAX
    oldYlogl = -REALMAX
    oldentropy = REALMAX
    oldAIC = REALMAX
    
    while 1:
        # apply EM steps to the complete mixture until convergence
        logging.debug('EM steps on complete mixture...')
        while 1:
            (W,M,R) = em_step(X,W,M,R,P)
            # likelihoods L (n x k) for all inputs and all components
            L = em_gauss(X,M,R)
            # mixture F (n x 1) and average log-likelihood
            F = L.dot(W)
            toosmall = len((F < REALMIN).nonzero()[0])
            F[F < REALMIN] = REALMIN
            logl = mean(log(F))
        
            # posteriors P (n x k) and their sums
            P = L * (ones((n,1)).dot(W.T)) / F.dot(ones((1,k)))
            
            if abs(logl/oldlogl-1) < THRESHOLD:
                logging.debug('Logl = %g' % (logl))
                break
            oldlogl = logl
    
        # for cross-validation, calculate loglikelihood for Y set
        if stop == 'cv':
            Fy = dot(em_gauss(Y,M,R),W)
            logging.debug('Fy nonzero: %d' % count_nonzero(F < REALMIN))
            Fy[Fy < REALMIN] = REALMIN
            Ylogl = mean(log(Fy))
            logging.debug('oldYlogl=%f Ylogl=%f' % (oldYlogl, Ylogl))
            if Ylogl < oldYlogl:
                if not converged:
                    converged = True
                    kconv = k
                    outf.tprint('# Calculation converged')
            if converged and k >= kconv+overfit:
                logging.debug('Ylogl decreased, stopping')
                outf.tprint('# Stopping due to Ylogl decrease')
                return (oldW, oldM, oldR, oldYlogl, Sk)
            oldYlogl = Ylogl
            oldW = W
            oldM = M
            oldR = R
        #
        
        newentropy = -logl-llcorrection
        if stop == 'cv':
            newentropy2 = -Ylogl-llcorrection
        logging.debug("with %d components S=%f J/K/mol (%f cal/K/mol)" % 
                      (k,8.314472*newentropy,1.9858775*newentropy))
        
        # calculate Akaike Information Criterion
        if stop == 'aicsd':
            npar = k-1 + d*k + k*d*(d+1)/2 # number of parameters of the mixture
            AIC = 2*npar - 2*n*(logl+llcorrection)
            logging.debug("AIC = %f" % (AIC))

        if stop == 'cv':
            newentropy2 = -Ylogl-llcorrection
            #Sk[k-1] = ufac*(newentropy+newentropy2)/2
            Sk[k-1] = ufac*newentropy
        else:
            Sk[k-1] = ufac*newentropy
            
        if toosmall:
            logging.warning(("%d out of %d likelihoods are too small. The following result may "+
                             "not be valid. Use fewer dimensions.") % (toosmall,len(F)))
                            
        if stop == 'aicsd': # use AIC+sdelta as stopping criterion
            if AIC > oldAIC:
                if not converged:
                    converged = True
                    kconv = k
                    outf.tprint('# Calculation converged by AIC')
            if converged and k >= kconv+overfit:
                logging.debug('Stopping as AIC has increased')
                logging.debug('%d %g' % (k,Sk[k-1]))
                outf.tprint('# Stopping due to AIC increase')
                return (oldW,oldM,oldR,logl,Sk) # return previous values
            oldW,oldM,oldR = W,M,R
                
        if printout:
            if stop == 'cv':
                outf.tprint("%d %g %g" % (k, Sk[k-1], ufac*newentropy2))
            else:
                outf.tprint("%d %g" % (k, Sk[k-1]))
        if stop == 'aicsd': # use sdelta as stopping criterion
            if abs(newentropy-oldentropy) < Sdelta/ufac:
                if not converged:
                    converged = True
                    kconv = k
                    outf.tprint('# Calculation converged by Sdelta')
            if converged and k >= kconv+overfit:
                logging.debug('Stopping %f %f' % (ufac*newentropy,ufac*oldentropy))
                outf.tprint('# Stopping due to reaching Sdelta')
                return (W,M,R,logl,Sk)
        oldentropy = newentropy
        if stop == 'aicsd':
            oldAIC = AIC
        
        # stop if kmax is reached
        if k == kmax:
            return (W,M,R,logl,Sk)
    
        # try to add another component
        logging.debug('Trying component allocation')
        ntries = 100 # repeat candidate search this many times maximum
        for itry in range(ntries):
            (Mnew,Rnew,alpha,returnstatus) = rand_split(P,X,M,R,sigma,F,W,nr_of_cand)
            if alpha != 0:
                break
            else:
                if returnstatus == 'split_impossible':
                    break
                elif itry < ntries-1:
                    logging.warning('No appropriate candidate found. Trying more candidates...')
        if alpha == 0:
            logging.warning('Failed to find candidates. Result has not converged.')
            if returnstatus == 'split_impossible':
                logging.warning('A larger sample is needed.')
            else:
                logging.warning('Try to use a larger ncand.')
            return (W,M,R,logl,Sk)
    
        K = em_gauss(X,Mnew,Rnew)
        PP = F*(1-alpha)+K*alpha
        LOGL = mean(log(PP))
    
        # optimize new mixture with partial EM steps updating only Mnew,Rnew
        veryoldlogl = logl
        oldlogl = LOGL
        done_here = False
      
        Pnew = (K*(ones((n,1))*alpha))/PP
        while not done_here:
            (alpha,Mnew,Rnew) = em_step(X,alpha,Mnew,Rnew,Pnew)
            K = em_gauss(X,Mnew,Rnew)
            Fnew = F*(1-alpha)+K*alpha
            Pnew = K*alpha/Fnew
            logl = mean(log(Fnew))
            if abs(logl/oldlogl-1) < THRESHOLD:
                done_here = True
            oldlogl = logl
    
        # check if log-likelihood increases with insertion
        if logl <= veryoldlogl:
            logging.debug(('Mixture uses only %d components, logl=%g oldlogl=%g veryoldlogl=%g '+
                          'llcorrection=%f') % (k, logl, oldlogl, veryoldlogl, llcorrection))
            return (W,M,R,logl,Sk)
        # allocate new component
        M = vstack((M,Mnew))
        R = vstack((R,Rnew))
        W = vstack(((1-alpha)*W, alpha))
        k = k + 1
        logging.debug(' k = %d' % (k))
        logging.debug('LogL = %g (%g after correction)' % (logl, logl+llcorrection))
        # prepare next EM step
        L = em_gauss(X,M,R)
        F = dot(L, W)
        F[F < REALMIN] = REALMIN
        P = L * dot(ones((n,1)),W.T) / dot(F,ones((1,k)))

        
# code to run if module is run from the command line

if __name__ == '__main__':

    import sys,time,argparse

    # calculate program CRC32 checksum
    f = open(__file__,'rb')
    b = f.read()
    f.close()
    cksum = ('%08x' % (crc32(b) & 0xffffffff)).upper()

    try:
        starttime = time.process_time() # only available in python 3.3+
    except:
        starttime = time.clock()
    
    # parse command line
    parser = argparse.ArgumentParser(
      description = 'Calculate entropy or mutual information by Gaussian mixture fitting.')
    parser.add_argument('-d',action='store_true',help='Turn on debug mode.')
    parser.add_argument(
      '-q',action='store_true',help='Quiet mode (no output to console)')
    parser.add_argument('-c', action='store_true', 
      help='Output only to console, do not write any files')
    parser.add_argument('-w',action='store_true',help='Overwrite existing output files')
    parser.add_argument('-J',default=None,help='Job number or job id string)')
    parser.add_argument(
      '--order',choices=['1', '1.5', '2', 'full'],default='full',
      help='Order of approximation (default: full)')
    parser.add_argument(
      '--center',action='store_true',
      help='Center distributions (assumes angle data in degrees) (default: no)')
    parser.add_argument('--centeronly',action='store_true',
      help='Only center and save data; do not calculate entropy')
    parser.add_argument('--stop', choices=['cv', 'aicsd'], default='cv',
      help='Stopping criterion to use: "cv" for cross-validation (default) or "aicsd" for '+
      'AIC & deltaS')
    parser.add_argument('--overfit', type=int, default=0,
      help='nr. of extra components to calculate after stopping crit. is met (default: 0)')
    parser.add_argument(
      '--sdelta',type=float, default=0.2, help='Calculation will stop when entropy change in '+
      'last step is below this value (default: 0.2) [only used with --stop aicsd]')
    parser.add_argument(
      '--emt',type=float,default=1e-5,
      help='Convergence threshold for EM calculation (default: 1e-5)')
    parser.add_argument(
      '--cols',default='all',
      help='Datafile columns to process as list of comma-separated ranges (default: all)')
    parser.add_argument('--slice', default='::', 
      help='Line selector slice, format: start:stop:step (default: "::")')
    parser.add_argument(
      '--maxk',type=int,default=200,
      help='Maximum number of components (default: 200)')
    parser.add_argument(
      '--ncand',type=int,default=30,
      help='Number of candidates to use in component search (default: 30)')
    parser.add_argument(
      '--unit',default='J',choices=['J','c','e'],
      help='Entropy unit to use: J for J/K/mol, c for cal/K/mol, e for nats. Default: J')
    parser.add_argument('--odir',default='.',help='Output directory (if not the current dir)')
    parser.add_argument('--version', action='version', 
      version='Program CRC32 hex checksum: '+cksum,
      help="Print the program's CRC32 checksum and exit")
    parser.add_argument('datafilename',help='The data file to process')
    a = parser.parse_args()

    # do not open output files if --centeronly option is specified
    if a.centeronly:
        a.c = True
    
    # entropy unit related variables
    unitdic = {'J':'J/K/mol','c':'cal/K/mol','e':'no'}
    unitfactordic = {'J':8.3144621,'c':1.9872041,'e':1.0}
    ufac = unitfactordic[a.unit]
    
    # file names
    fnroot = a.odir+'/'+a.datafilename[:-4]+'.gme'
    if a.J == None:
        jobno = ''
    else:
        jobno = '.'+a.J
        #a.q = True
    outputfilename = fnroot+jobno+'.out'
    logfilename = fnroot+jobno+'.log'
    npzfilename = fnroot+jobno+'.npz'
    if not a.c and not a.w and (os.path.exists(outputfilename) or os.path.exists(logfilename) or 
      os.path.exists(npzfilename)):
        logging.error('Output file(s) already exist. Use -w to overwrite them.')
        sys.exit(1)
    if a.c:
        fout = sys.stdout
        flog = sys.stdout
    else:
        fout = open(outputfilename,'w',1)
        flog = open(logfilename,'w',1)
    if a.q:
        outf = tee(fout,flog)
    elif a.c:
        outf = tee(sys.stdout)
    else:
        outf = tee(sys.stdout,fout,flog)

    # set logging
    loglevel = logging.DEBUG if a.d else logging.WARNING
    logging.basicConfig(format='# %(levelname)s: %(message)s',level=loglevel,stream=flog)
    if not a.q and not a.c:
        console = logging.StreamHandler()
        console.setLevel(loglevel)
        formatter = logging.Formatter('# %(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # print parameters
    outf.tprint("# GMENTROPY by Andras Szilagyi")
    outf.tprint("# Program CRC32 hex checksum: %s" % (cksum))
    if not a.centeronly:
        outf.tprint("# Calculating entropy to %s order" % (a.order))
        if a.stop=='cv' and a.maxk > 1:
            outf.tprint('# Cross-validation will be used as stopping criterion')
        elif a.maxk > 1:
            outf.tprint('# AIC and entropy change will be used as stopping criterion')
            outf.tprint("# sdelta - entropy convergence criterion: %g" % a.sdelta)
        elif a.maxk == 1:
            outf.tprint('# Quasiharmonic calculation (k=1) will be performed')
        outf.tprint("# emt - EM convergence criterion: %g" % a.emt)
    outf.tprint("# Distribution centering will be"+(' ' if a.center or a.centeronly 
      else ' not ')+"performed")
    outf.tprint("# Data file column selection: "+a.cols)
    if a.slice != '::':
        outf.tprint("# Line selector slice: %s" % a.slice)
    if not a.centeronly:
        outf.tprint("# maxk - maximum number of components: %d" % a.maxk)
        outf.tprint("# ncand - number of candidates during component search: %d" % a.ncand)
        outf.tprint("# Entropy and mutual information reported in "+unitdic[a.unit]+" units")
    outf.tprint("# Input data file: "+a.datafilename)
    if not a.centeronly:
        if a.c:
            outf.tprint('# Output to console only')
        else:
            outf.tprint('# Output directory: %s' % (a.odir))
            outf.tprint('# Output file: %s' % (outputfilename))
            outf.tprint('# Log file: %s' % (logfilename))
            outf.tprint('# Mixture parameters will be saved to: %s' % (npzfilename))
        if a.w:
            outf.tprint("# Existing output files will be overwritten")
    outf.tprint('# Loading data...')
    
    # load data
    Xfull = loadtxt(a.datafilename)
    
    if len(Xfull.shape) == 1:
        Xfull = Xfull.reshape((-1,1))
    
    # select lines and columns if specified on command line
    Xfull = eval('Xfull['+a.slice+',:]')
    if a.cols != 'all':
        selcol = array(klist(a.cols))-1
        Xfull = Xfull[:,selcol]
    
    nfull,d = Xfull.shape
    
        
    # center distributions if requested
    if a.center or a.centeronly:
        tompipi(Xfull) # shift angles to [-180,180) if needed
        Xfull = center_angle_distribution(Xfull)
        if a.centeronly: # save centered data and exit
            cdatafilename = a.datafilename[:-3]+'centered'+a.datafilename[-4:]
            savetxt(cdatafilename, Xfull)
            print('Centered data saved to %s' % cdatafilename)
            sys.exit()
            
    # for cross-validation, divide data set randomly in two halves
    if a.stop=='cv' and a.maxk > 1:
        nra.shuffle(Xfull)
        half = int(nfull/2)
        X = Xfull[:half, :]
        Y = Xfull[half:, :]
        outf.tprint('# Dividing data set in two halves for cross-validation')
    else:
        X = Xfull
        Y = zeros((nfull, d))
    #
    
    n,d = X.shape
    
    outf.tprint('# Data matrix dimensions: %d x %d' % (n,d))

    if a.order=='full': # full-order entropy calculation
        outf.tprint('#')
        outf.tprint('# k entropy')
        (W, M, R, Tlogl, Sk) = em(X, Y, a.maxk, a.ncand, a.stop, a.sdelta, a.emt, ufac, 
	  a.overfit, printout=True)
    
        # save component data to .npz file
        if not a.c:
            savez(npzfilename, W=W, M=M, R=R) 
    else: # lower order calculations
        outf.tprint("#")
        outf.tprint("# 1-dimensional marginal entropies (variable number, entropy, number "+
          "of components)")
        marg1d=zeros(d)
        for i in range(d):
            (W, M, R, Tlogl, Sk) = em(X[:, i].reshape((-1, 1)), Y[:, i].reshape((-1, 1)),
	      a.maxk, a.ncand, a.stop, a.sdelta, a.emt, ufac, a.overfit, printout=False)
            k = len(W)
            outf.tprint('%d %g %d' % (i+1, Sk[k-1,0], k))
            marg1d[i]=Sk[k-1, 0]
        S1 = marg1d.sum()
        outf.tprint("# First-order entropy (sum of 1D marginal entropies):")
        outf.tprint("%f" % (S1))
        if a.order in ['1.5', '2']: # first-order plus correlations
            corrmat = corrcoef(X, rowvar=0)
            estmi = 0.0 # mutual info estimated from corr coef
            for i in range(d-1):
                for j in range(i+1, d):
                    estmi = estmi-0.5*log(1.0-corrmat[i, j]**2)
            outf.tprint('# Estimated mutual info from linear correlations:')
            outf.tprint('%f' % (estmi*ufac))
            # also calculate "quasi-harmonic mutual info"
            CX = cov(Xfull, rowvar=0)
            Iqh = 0.5*(log(diag(CX)).sum()-log(la.det(CX)))
            outf.tprint('# Quasiharmonic mutual info:')
            outf.tprint('%f' % (Iqh*ufac))
            outf.tprint('# First-order entropy corrected by estimated / quasiharm. mut.inf')
            outf.tprint('%f %f' % (S1-estmi*ufac, S1-Iqh*ufac))
        if a.order == '2':
            I2 = 0.0
            outf.tprint("# 2-dimensional joint (marginal) entropies")
            outf.tprint("# variable 1, variable 2, joint entropy, number of components, "+
              "mutual information")
            for i in range(d-1):
                for j in range(i+1,d):
                    (W, M, R, Tlogl, Sk) = em(X[:, [i, j]],Y[:, [i, j]], a.maxk, a.ncand,
                      a.stop, a.sdelta, a.emt, ufac, a.overfit, printout=False)
                    k = len(W)
                    Iij = marg1d[i]+marg1d[j]-Sk[k-1, 0]
                    outf.tprint('%d %d %g %d %g' % (i+1, j+1, Sk[k-1,0], k, Iij))
                    I2 += Iij
            outf.tprint("# Sum of pairwise mutual information values:")
            outf.tprint("%f" % I2)
            outf.tprint('# Mutual info from supralinear correlations:')
            outf.tprint('%f' % (I2-estmi*ufac))
            outf.tprint("# Second-order entropy (first-order entropy minus sum of pairwise "+
              "mutual informations):")
            outf.tprint("%f" % (S1-I2))
    try:
        endtime = time.process_time()
        ttype = '(cpu time)'
    except:
        endtime = time.clock()
        ttype = '(real time)'
    outf.tprint("# Execution time %s in seconds: %d" % (ttype, round(endtime-starttime)))

