
from utils import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wurlitzer import sys_pipes
import sys
from multiprocessing.pool import ThreadPool

import numpy as np
U = 1 # upper limit
N = 10000
epsilon = 1 # reset parameter
deltaU = 0.022
alpha = 1 - 1/np.sqrt(N)

def ehe_critical_weights(N,U=1):
    return np.ones([N,N])*(1 - 1/np.sqrt(N)) * U / N

def simulate_model(units=np.random.random(N)*U,numAvalanches=1000,
                   W=None,deltaU = deltaU):
    if W is None:
        W = ehe_critical_weights(len(units))
    avalanche_sizes = []
    avalanche_durations = []
    sumavd = 0
    while sumavd < numAvalanches:
        r = np.random.randint(len(units))
        units[r] += deltaU
        if units[r] >= U:
            avs,avd = handle_avalanche(r,units,W,epsilon=1,U=U)
            avalanche_sizes.append(avs)
            avalanche_durations.append(avd)
            sumavd += avd
    return avalanche_sizes,avalanche_durations

def handle_avalanche(start_unit,units,W,epsilon=1,U=U):
    
    avalanche_size = 0
    avalanche_duration = 0
    # handle starting single unit avalanche
    if start_unit is None: 
        A = units >= U
        units[A] = epsilon * (units[A] - U)
        s = np.sum(A)
    else:
        A = np.zeros_like(units)
        A[start_unit] = 1
        units[start_unit] = epsilon * (units[start_unit] - U)
        s = 1
    while s > 0:
        avalanche_size += s
        avalanche_duration += 1
        units += W @ A #interior input to all units
        A = units >= U
        units[A] = epsilon * (units[A] - U) # resetting threshold crossed units
        s = np.sum(A)
    return avalanche_size,avalanche_duration

ehe_module = load_module('ehe_detailed')

from functools import reduce
import itertools

def spiking_configurations(iterable):
    "powerset([1,2,3]) --> reverse of (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s),0,-1))
  
def intersection(hyperr1,hyperr2):
    """compute intersection of two hyperrectangles given by list of  end values along each axes.
       So the rectangles always have their lower values at 0"""
    if len(hyperr1) == 0 or len(hyperr2) == 0:
        return []
    intersection  = []
    for (u1l,u1u),(u2l,u2u) in zip(hyperr1,hyperr2):
        il,iu = (max(u1l,u2l),min(u1u,u2u))
        if iu <= il:
            return []
        intersection.append((il,iu))
    return intersection

def volume(hyperr):
    if len(hyperr) == 0:
        return 0
    return prod(hu - hl for hu,hl in hyperr)

def volume_union(hyperrectangles):
    # uses approach from https://stackoverflow.com/questions/28146260
    # TODO use best known algorithm for Klee's measure problem
    # https://pdfs.semanticscholar.org/9528/ef345d013577fd6c9d7fde8a54a1b86d3bbe.pdf
    dim = len(hyperrectangles[0])
    gridlines = [np.unique([hyperr[i][0] for hyperr in hyperrectangles] +
                           [hyperr[i][1] for hyperr in hyperrectangles]) for i in range(dim)]
    grid = np.zeros([len(gl)-1 for gl in gridlines],dtype=bool) #TODO Better to create grid on the fly to save space?
    #fill the grid
    for hyperr in hyperrectangles:
        grid_slices = tuple(slice(np.searchsorted(gridlines[i],hyperr[i][0]),np.searchsorted(gridlines[i],hyperr[i][1]))
                            for i in range(dim))
        grid[grid_slices] = True #indicate grid cells covered by hyperr
    # calculate total volume from grid and gridlines
    it = np.nditer(grid, flags=['multi_index'])
    volume = 0
    while not it.finished:
        if(it[0]): # for each filled cell compute it's volume and add it
            volume += prod(grid[i+1] - grid[i] for grid,i in zip(gridlines,it.multi_index))
        it.iternext()
    return volume

def R_sp(W,spiking_configuration,num_dims):
    """returns excluded hyper rectangle induced by the spiking_configuration"""
    ind_sp = np.zeros(num_dims,dtype=int)
    for s in spiking_configuration:
        ind_sp[s] = 1
    limits = (W @ ind_sp) # Don't rescale here
    sp_comp = np.ones_like(ind_sp) - ind_sp # 1 at nodes that are not in sp and 0 otherwise
    return [(0,x) for x in np.maximum(limits,sp_comp)]

def noninhabited_region(units,W,N=None):
    if N is None:
      N = len(units)
    return volume_union([R_sp(W,sp,N) for sp in spiking_configurations(units)])

from itertools import chain, combinations

def prod(gen):
    res = 1;
    for i in gen:
        res *= i
    return res

def nonempty_subsets(iterable):
    "nonempty_subsets([1,2,3]) -->  (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)))

class NoninhabitedRegion():
    def __init__(self,W):
        self.results = {}
        self.W = W;

    def noninhabited_region(self,N=None,U=None):
        W = self.W
        if N is None:
            N = range(W.shape[0])
        elif type(N) == int:
            N = range(N)
        N = tuple(N)
        if U is None:
            U = np.ones(len(N))
        U = np.array(U)
        res = self.results.get((N,tuple(U)))
        if res is not None:
            res
        row_sums = np.array([sum(W[i,j] for j in N) for i in N]) # works with sympy
        res = prod(row_sums)#Gamma with I=N
        for I in nonempty_subsets(N):
            I_inds = np.in1d(N,I)
            res += self.noninhabited_region(I,U=row_sums[I_inds]) * prod(U[~I_inds]-row_sums[~I_inds])
        self.results[(N,tuple(U))] = res
        return res

import sympy as sym
def symbolic_matrix(N):
     W = np.empty((N,N),dtype=object);
     for i in range(N):
         for j in range(N):
             W[i,j] = sym.symbols('w'+str(i)+str(j))
     return W

from sympy.combinatorics import Permutation
def symbolic_det(W,I):
    ret = 0
    for perm_idx in itertools.permutations(range(len(I))):
        ret += Permutation(perm_idx).signature() * prod(W[i,I[j_idx]] for i,j_idx in zip(I,perm_idx))
    return ret

def hypothesis_volume(W,N):
    if type(N) == int:
        N = range(N)
    return sum((-1 if len(I) % 2 == 0 else 1) * symbolic_det(W,I) for I in spiking_configurations(N))

def test_conjecture(N=5):
    W = symbolic_matrix(N)
    NR = NoninhabitedRegion(W)
    Lambda = NR.noninhabited_region()
    print('Lambda',Lambda,flush=True)
    Lambda = sym.simplify(Lambda)
    Lambda_bar = hypothesis_volume(W,N)
    print('Lambda_bar',Lambda_bar,flush=True)
    Lambda_bar = sym.simplify(Lambda_bar)
    print(sym.simplify(Lambda - Lambda_bar))

from sympy import Symbol, Rational, binomial, expand_func
from sympy.utilities.iterables import partitions

def gen_partitions(N):
    for p in partitions(N,m=N):
        yield reduce(lambda a,b:a+b,([k]*v for k,v in p.items()))


def coef_i(p,i,N):
    return sum((-1)**(len(J) - 1)*binomial(N - sum(J),i - sum(J)) for J in (list(nonempty_subsets(p)) + [p]))

def coef(p,N):
    return sum((-1)**(N-i)*coef_i(p,i,N) for i in range(1,N+1))


def sign(p,N):
    if sum(p) != N:
        return 0;
    return (-1)**(sum(lz-1 for lz in p))

def expected_sign(p,N): 
    return (-1)**(N-1)*sign(p,N)

from itertools import chain, combinations
import sympy as sym
class AvalancheSizeDistribution():
    def __init__(self,W,deltaU=0.022,symbolic=False):
        self.W = W
        self.N = W.shape[0]
        self.deltaU = deltaU
        self.volumes = {}
        self.detailed_volumes = {}
        self.probs = {}
        self.det_func = np.linalg.det \
                        if not symbolic else lambda a: sym.Matrix(a).det()

    def detailed_volume(self,i_s,J):
        J = frozenset(J)
        vol = self.detailed_volumes.get((i_s,J))
        if vol is not None:
            return vol
        # Calculate volume of avalanche size with determinant formula
        det = self.det_func
        rem_u = list(J-set([i_s]))
        Jc = list(set(range(self.N))-J)
        vol_is = self.deltaU
        vol_J = 1
        if len(rem_u) > 0:
            m = np.diag([np.sum(self.W[i,list(J)]) for i in rem_u]) \
                                 - self.W[np.ix_(rem_u,rem_u)]
            vol_J = det(m) 
        vol_Jc = 1
        if len(Jc) > 0:
            m = np.diag([1 - np.sum(self.W[i,list(J)]) for i in Jc]) \
                                    - self.W[np.ix_(Jc,Jc)]
            vol_Jc = det(m)
        vol = vol_is*vol_J*vol_Jc
        self.detailed_volumes[(i_s,J)] = vol
        return vol

    def volume_size0(self):
        vol = 0
        for i in range(self.N):
            inds = list(set(range(self.N))-set([i]))
            vol += self.deltaU*self.det_func(np.eye(self.N-1,dtype=int) - self.W[np.ix_(inds,inds)])
        return vol

    def volume(self,n):
        vol = self.volumes.get(n)
        if vol is not None:
            return vol
        if n == -1:
            vol = self.noninh_volume()
            self.volumes[n] = vol
            return vol
        if n == 0:
            vol = self.volume_size0()
            self.volumes[n] = vol
            return vol
        vol = 0
        for units in combinations(range(self.N),n):
            for i_s in units:
                vol += self.detailed_volume(i_s,set(units))
        self.volumes[n] = vol
        return vol

    def prob(self,n):
        prob = self.probs.get(n)
        if prob is not None:
            return prob
        norm_n = self.volume(n)/self.N # 1/N prob to get activation to correct starting unit
        norm_0 = self.volume(0)/self.N
        prob = norm_n/(norm_0)
        self.probs[n] = prob
        return prob

    def avs_dist(self):
        for i in range(1,self.N+1):
            self.prob(i)
        return self.probs

from scipy.special import binom

def vnonin2ovl(ns1,ns2,no,Us1,Us2,Uo,alpha):
    #TODO compute in log space
    return alpha * ((ns1*Us1**(ns1-1)*Uo**no*Us2**ns2 if ns1>0 else 0) +  
                   (no*Us1**(ns1)*Uo**(no-1)*Us2**ns2 if no>0 else 0) + 
                   (ns2*Us1**ns1*Uo**no*Us2**(ns2-1) if ns2>0 else 0)) -  \
           alpha**2 * (ns1*ns2*Us1**(ns1-1)*Uo**no*Us2**(ns2-1) if ns1*ns2>0 else 0) - \
           alpha**3 * (ns1*no*ns2*Us1**(ns1-1)*Uo**(no-1)*Us2**(ns2-1) if ns1*no*ns2>0 else 0)

import numpy as np
def Wovl(ns1,ns2,no,alpha,dtype=float):
    W = np.zeros((ns1+ns2+no,ns1+ns2+no),dtype=dtype)
    inds1 = range(ns1+no)
    W[np.ix_(inds1,inds1)] = alpha
    inds2 = range(ns1,ns1+no+ns2)
    W[np.ix_(inds2,inds2)] = alpha
    return W

from itertools import chain, combinations
import sympy as sym
class Simple2ovlAvalancheSizeDistributionFast():
    def __init__(self,Ns1,Ns2,No,alpha,deltaU=0.022,symbolic=False):
        self.Ns1 = Ns1
        self.Ns2 = Ns2
        self.No = No
        self.alpha = alpha
        self.N = self.Ns1+self.Ns2+self.No
        self.W = Wovl(Ns1,Ns2,No,alpha,dtype=object if symbolic else float)
        self.deltaU = deltaU
        self.volumes = {}
        self.detailed_volumes = {}
        self.probs = {}
        self.det_func = np.linalg.det \
                        if not symbolic else lambda a: sym.Matrix(a).det()


    def detailed_volume(self,ns1,ns2,no):
        vol = self.detailed_volumes.get((ns1,ns2,no))
        if vol is not None:
            return vol
        # Calculate volume of avalanche size with determinant formula
        vol_is = self.deltaU
        a = self.alpha
        n = ns1+ns2+no
        U1,U2,Uo = (ns1+no)*a,(ns2+no)*a,n*a
        #print(ns1,ns2,no,U1,U2,Uo)
        if n > 1:
             vol = 0
             if ns1 > 0:
                 #starting unit in s1
                 vol += ns1 * (U1**(ns1-1)*Uo**no*U2**ns2 \
                               - vnonin2ovl(ns1-1,ns2,no,U1,U2,Uo,a))
             if ns2 > 0:
                 #starting unit in s1
                 vol += ns2 * (U1**ns1*Uo**no*U2**(ns2-1)\
                               - vnonin2ovl(ns1,ns2-1,no,U1,U2,Uo,a))
             if no > 0:
                 #starting unit in s1
                 vol += no * (U1**ns1*Uo**(no-1)*U2**ns2 \
                              - vnonin2ovl(ns1,ns2,no-1,U1,U2,Uo,a))
        else:
            vol = 1                     
        volc = 1
        Uc1,Uc2,Uco = 1-U1,1-U2,1-Uo
        if n < self.N:
            volc = (Uc1**(self.Ns1-ns1)*Uco**(self.No-no)*Uc2**(self.Ns2-ns2) \
                    - vnonin2ovl((self.Ns1-ns1),(self.Ns2-ns2),(self.No-no),Uc1,Uc2,Uco,a))
        vol = vol_is*vol*volc
        self.detailed_volumes[(ns1,ns2,no)] = vol
        return vol

    def volume_size0(self):
        vol = 0
        Ns1,Ns2,No = self.Ns1,self.Ns2,self.No
        alpha = self.alpha
        vol += Ns1*self.deltaU*(1-vnonin2ovl(Ns1-1,Ns2,No,1,1,1,alpha))
        vol += Ns2*self.deltaU*(1-vnonin2ovl(Ns1,Ns2-1,No,1,1,1,alpha))                    
        vol += No*self.deltaU*(1-vnonin2ovl(Ns1,Ns2,No-1,1,1,1,alpha))
        return vol

    def volume(self,n):
        vol = self.volumes.get(n)
        if vol is not None:
            return vol
        print("n",n)    
        if n == 0:
            vol = self.volume_size0()
            self.volumes[n] = vol
            return vol
        if n == -1:
            vol = self.noninh_volume()
            self.volumes[n] = vol
            return vol
        vol = 0
        for ns1 in range(min(self.Ns1+1,n+1)):
            for ns2 in range(min(self.Ns2+1,n-ns1+1)):
                no = n - ns1 - ns2
                if no <= self.No:
                   #TODO compute this log based for larger matrices
                   num_ass = binom(self.Ns1,ns1)*binom(self.Ns2,ns2)*binom(self.No,no) 
                   vol += num_ass*self.detailed_volume(ns1,ns2,no)
        self.volumes[n] = vol
        return vol

    def prob(self,n):
        prob = self.probs.get(n)
        if prob is not None:
            return prob
        norm_n = self.volume(n)/self.N # 1/N prob to get activation to correct
        norm_0 = self.volume(0)/self.N
        # norm_noninh = self.volume(-1)
        # prob = norm_n/(norm_noninh - norm_0)
        prob = norm_n/norm_0
        self.probs[n] = prob
        return prob

    def avs_dist(self):
        for i in range(1,self.N+1):
            self.prob(i)
        return self.probs

from itertools import chain, combinations
import sympy as sym
class Simple2ovlAvalancheSizeDistribution():
    def __init__(self,Ns1,Ns2,No,alpha,deltaU=0.022,symbolic=False):
        self.Ns1 = Ns1
        self.Ns2 = Ns2
        self.No = No
        self.alpha = alpha
        self.N = self.Ns1+self.Ns2+self.No
        self.W = Wovl(Ns1,Ns2,No,alpha,dtype=object if symbolic else float)
        self.deltaU = deltaU
        self.volumes = {}
        self.detailed_volumes = {}
        self.probs = {}
        self.det_func = np.linalg.det \
                        if not symbolic else lambda a: sym.Matrix(a).det()


    def detailed_volume(self,i_s,J):
        J = frozenset(J)
        #print('i_s','J',i_s,J)
        vol = self.detailed_volumes.get((i_s,J))
        if vol is not None:
            return vol
        # Calculate volume of avalanche size with determinant formula
        det = self.det_func
        rem_u = list(J-set([i_s]))
        Jc = list(set(range(self.N))-J)
        vol_is = self.deltaU
        vol_J = 1
        if len(rem_u) > 0:
            m = np.diag([np.sum(self.W[i,list(J)]) for i in rem_u]) \
                                 - self.W[np.ix_(rem_u,rem_u)]
            #print('m',m)
            vol_J = det(m) 
            #print('vol_J',vol_J)
        vol_Jc = 1
        if len(Jc) > 0:
            m = np.diag([1 - np.sum(self.W[i,list(J)]) for i in Jc]) \
                                    - self.W[np.ix_(Jc,Jc)]
            #print('mc',m)                        
            vol_Jc = det(m)
            #print('vol_Jc',vol_Jc)
        vol = vol_is*vol_J*vol_Jc
        self.detailed_volumes[(i_s,J)] = vol
        return vol


    def volume_size0(self):
        vol = 0
        for i in range(self.N):
            m = np.diag([1 if j!=i else 1-self.deltaU for j in range(self.N)]) -self.W
            vol = vol + self.det_func(m)
        return vol

    def noninh_volume(self):
        return self.det_func(np.eye(self.N,dtype=int)-self.W)

    def volume(self,n):
        vol = self.volumes.get(n)
        if vol is not None:
            return vol
        #print("n",n)    
        if n == -1:
            vol = self.noninh_volume()
            self.volumes[n] = vol
            return vol
        if n == 0:
            vol = self.volume_size0()
            self.volumes[n] = vol
            return vol
        vol = 0
        for ns1 in range(min(self.Ns1+1,n+1)):
            for ns2 in range(min(self.Ns2+1,n-ns1+1)):
                no = n - ns1 - ns2
                if no <= self.No:
                   num_ass = binom(self.Ns1,ns1)*binom(self.Ns2,ns2)*binom(self.No,no) 
                   units = list(range(ns1))+list(range(self.Ns1,self.Ns1+no))+\
                           list(range(self.Ns1+self.No,self.Ns1+self.No+ns2))
                   for i_s in units:
                        vol += num_ass*self.detailed_volume(i_s,set(units))
        self.volumes[n] = vol
        return vol

    def prob(self,n):
        prob = self.probs.get(n)
        if prob is not None:
            return prob
        norm_n = self.volume(n)/self.N # 1/N prob to get activation to correct
        norm_0 = self.volume(0)/self.N
        norm_noninh = self.volume(-1)

        prob = norm_n/(norm_noninh - norm_0)
        self.probs[n] = prob
        return prob

    def avs_dist(self):
        for i in range(1,self.N+1):
            self.prob(i)
        return self.probs

from itertools import chain, combinations
import sympy as sym

def W_mod2ovl(ns1,ns2,alpha,alpha_cross,dtype=float):
    W = np.zeros((ns1+ns2,ns1+ns2),dtype=dtype)
    inds1 = list(range(ns1))
    inds2 = list(range(ns1,ns1+ns2))
    W[np.ix_(inds1,inds1)] = alpha
    W[np.ix_(inds2,inds2)] = alpha
    W[np.ix_(inds2,inds1)] = alpha_cross
    W[np.ix_(inds1,inds2)] = alpha_cross
    return W


class ModSimple2ovlAvalancheSizeDistributionDet():
    def __init__(self,Ns1,Ns2,alpha,alpha_cross,deltaU=0.022,symbolic=False):
        self.Ns1 = Ns1
        self.Ns2 = Ns2
        self.alpha_cross = alpha_cross
        self.alpha = alpha
        self.N = self.Ns1+self.Ns2
        self.W = W_mod2ovl(Ns1,Ns2,alpha,alpha_cross,dtype=object if symbolic else float)
        self.deltaU = deltaU
        self.volumes = {}
        self.detailed_volumes = {}
        self.probs = {}
        self.det_func = np.linalg.det \
                        if not symbolic else lambda a: sym.Matrix(a).det()


    def detailed_volume(self,i_s,J):
        J = frozenset(J)
        #print('i_s','J',i_s,J)
        vol = self.detailed_volumes.get((i_s,J))
        if vol is not None:
            return vol
        # Calculate volume of avalanche size with determinant formula
        det = self.det_func
        rem_u = list(J-set([i_s]))
        Jc = list(set(range(self.N))-J)
        vol_is = self.deltaU
        vol_J = 1
        if len(rem_u) > 0:
            m = np.diag([np.sum(self.W[i,list(J)]) for i in rem_u]) \
                                 - self.W[np.ix_(rem_u,rem_u)]
            #print('m',m)
            vol_J = det(m) 
            #print('vol_J',vol_J)
        vol_Jc = 1
        if len(Jc) > 0:
            m = np.diag([1 - np.sum(self.W[i,list(J)]) for i in Jc]) \
                                    - self.W[np.ix_(Jc,Jc)]
            #print('mc',m)                        
            vol_Jc = det(m)
            #print('vol_Jc',vol_Jc)
        vol = vol_is*vol_J*vol_Jc
        self.detailed_volumes[(i_s,J)] = vol
        return vol


    def volume_size0(self):
        vol = 0
        for i in range(self.N):
            m = np.diag([1 if j!=i else 1-self.deltaU for j in range(self.N)]) -self.W
            vol = vol + self.det_func(m)
        return vol

    def noninh_volume(self):
        return self.det_func(np.eye(self.N,dtype=int)-self.W)

    def volume(self,n):
        vol = self.volumes.get(n)
        if vol is not None:
            return vol
        #print("n",n)    
        if n == -1:
            vol = self.noninh_volume()
            self.volumes[n] = vol
            return vol
        if n == 0:
            vol = self.volume_size0()
            self.volumes[n] = vol
            return vol
        vol = 0
        for ns1 in range(min(self.Ns1+1,n+1)):
            ns2 = n - ns1
            if ns2 <= self.Ns2:
                   num_ass = binom(self.Ns1,ns1)*binom(self.Ns2,ns2) 
                   units = list(range(ns1))+list(range(self.Ns1,self.Ns1+ns2))
                   for i_s in units:
                        vol += num_ass*self.detailed_volume(i_s,set(units))
        self.volumes[n] = vol
        return vol

    def prob(self,n):
        prob = self.probs.get(n)
        if prob is not None:
            return prob
        norm_n = self.volume(n)/self.N # 1/N prob to get activation to correct
        norm_0 = self.volume(0)/self.N
        norm_noninh = self.volume(-1)

        prob = norm_n/(norm_noninh - norm_0)
        self.probs[n] = prob
        return prob

    def avs_dist(self):
        for i in range(1,self.N+1):
            self.prob(i)
        return self.probs

from scipy.special import binom

def vnoninmod2ovl(ns1,ns2,Us1,Us2,alpha,alpha_cross):
    #TODO compute in log space
    return alpha * ((ns1*Us1**(ns1-1)*Us2**ns2 if True or ns1>0 else 0) + (ns2*Us1**ns1*Us2**(ns2-1) if True or ns2>0 else 0)) - \
           (alpha**2 - alpha_cross**2) * (ns1*ns2*Us1**(ns1-1)*Us2**(ns2-1) if True or ns1*ns2 > 0 else 0)
    # return alpha * ((ns1*Us1**(ns1-1)*Uo**no*Us2**ns2 if ns1>0 else 0) +  
    #                (no*Us1**(ns1)*Uo**(no-1)*Us2**ns2 if no>0 else 0) + 
    #                (ns2*Us1**ns1*Uo**no*Us2**(ns2-1) if ns2>0 else 0)) -  \
    #        alpha**2 * (ns1*ns2*Us1**(ns1-1)*Uo**no*Us2**(ns2-1) if ns1*ns2>0 else 0) - \
    #        alpha**3 * (ns1*no*ns2*Us1**(ns1-1)*Uo**(no-1)*Us2**(ns2-1) if ns1*no*ns2>0 else 0)

from itertools import chain, combinations
import sympy as sym
class SimpleMod2ovlAvalancheSizeDistributionFast():
    def __init__(self,Ns1,Ns2,alpha,alpha_cross,deltaU=0.022,symbolic=False):
        self.Ns1 = Ns1
        self.Ns2 = Ns2
        self.alpha_cross = alpha_cross
        self.alpha = alpha
        self.N = self.Ns1+self.Ns2
        self.W = W_mod2ovl(Ns1,Ns2,alpha,alpha_cross,dtype=object if symbolic else float)
        self.deltaU = deltaU
        self.volumes = {}
        self.detailed_volumes = {}
        self.probs = {}
        self.det_func = np.linalg.det \
                        if not symbolic else lambda a: sym.Matrix(a).det()


    def detailed_volume(self,ns1,ns2):
        vol = self.detailed_volumes.get((ns1,ns2))
        if vol is not None:
            return vol
        # Calculate volume of avalanche size with determinant formula
        vol_is = self.deltaU
        a = self.alpha
        ac = self.alpha_cross
        U1,U2= ns1*a+ns2*ac,ns2*a+ns1*ac
        if ns1+ns2 > 1:
             vol = 0
             if ns1 > 0:
                 #starting unit in s1
                 vol += ns1*(U1**(ns1-1)*U2**ns2 \
                               - vnoninmod2ovl(ns1-1,ns2,U1,U2,a,ac))
             if ns2 > 0:
                 #starting unit in s1
                 vol += ns2*(U1**ns1*U2**(ns2-1)\
                               - vnoninmod2ovl(ns1,ns2-1,U1,U2,a,ac))
        else:
            vol = 1                     
        volc = 1
        Uc1,Uc2 = 1-U1,1-U2
        if ns1+ns2 < self.N:
            volc = (Uc1**(self.Ns1-ns1)*Uc2**(self.Ns2-ns2) \
                    - vnoninmod2ovl((self.Ns1-ns1),(self.Ns2-ns2),Uc1,Uc2,a,ac))
        vol = vol_is*vol*volc
        self.detailed_volumes[(ns1,ns2)] = vol
        return vol

    def volume_size0(self):
        vol = 0
        Ns1,Ns2 = self.Ns1,self.Ns2
        vol += Ns1*self.deltaU*(1-vnoninmod2ovl(Ns1-1,Ns2,1,1,self.alpha,self.alpha_cross))
        vol += Ns2*self.deltaU*(1-vnoninmod2ovl(Ns1,Ns2-1,1,1,self.alpha,self.alpha_cross))                    
        return vol

    def volume(self,n):
        vol = self.volumes.get(n)
        if vol is not None:
            return vol
        print("n",n)    
        if n == 0:
            vol = self.volume_size0()
            self.volumes[n] = vol
            return vol
        # if n == -1:
        #     vol = self.noninh_volume()
        #     self.volumes[n] = vol
        #     return vol
        vol = 0
        for ns1 in range(min(self.Ns1+1,n+1)):
            ns2 = n - ns1
            if ns2 <= self.Ns2:
                   #TODO compute this log based for larger matrices
                   num_ass = binom(self.Ns1,ns1)*binom(self.Ns2,ns2)
                   vol += num_ass*self.detailed_volume(ns1,ns2)
        self.volumes[n] = vol
        return vol

    def prob(self,n):
        prob = self.probs.get(n)
        if prob is not None:
            return prob
        norm_n = self.volume(n)/self.N # 1/N prob to get activation to correct
        norm_0 = self.volume(0)/self.N
        # norm_noninh = self.volume(-1)
        # prob = norm_n/(norm_noninh - norm_0)
        prob = norm_n/norm_0
        self.probs[n] = prob
        return prob

    def avs_dist(self):
        for i in range(1,self.N+1):
            self.prob(i)
        return self.probs

def avs_dist_nk(Ns,a,ac,ns1,ns2):
    if ns1+ns2 == 0:
       return (0,0)
    s1,s2,l1,l2 = ns1*a+(ns2)*ac,(ns2)*a+ns1*ac,Ns-ns1,Ns-(ns2)
    S1,S2 = 1-s1,1-s2
    mult = binom(Ns,ns1)*binom(Ns,ns2)
    if mult <= 0:
        return (0,0)
    ret = mult*ac*s1**ns1*s2**(ns2)*S1**(l1-1)*S2**(l2-1)*((S1-(a*l1))*(S2-(a*l2))-(l1*l2)*ac**2) / \
                     (ns1*(ns2)*(a**2+ac**2)+a*ac*(ns1**2+(ns2)**2)) 
    ret = ret/(2*Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))
    return (ret*ns1,ret*(ns2))
 
def avs_distmod2ovl(Ns,a,ac):
    res = []
    for n in range(1,2*Ns+1):
        probn = 0
        probn_log = 0
        for k in range(n+1):
            s1,s2,l1,l2 = k*a+(n-k)*ac,(n-k)*a+k*ac,Ns-k,Ns-(n-k)
            S1,S2 = 1-s1,1-s2
            mult = binom(Ns,k)*binom(Ns,n-k)
            pos = lambda x: x
            if mult >= 0:
               add = mult*n*ac*s1**k*s2**(n-k)*S1**(l1-1)*S2**(l2-1)*((S1-pos(a*l1))*(S2-pos(a*l2))-pos(l1*l2)*ac**2) / \
                      (k*(n-k)*(a**2+ac**2)+a*ac*(k**2+(n-k)**2))
               #import pdb;pdb.set_trace()       
               probn+=add 
            # multlog = log2comb(Ns,k)+log2comb(Ns,n-k)
            # if multlog >= 0:
            #     addlog =((multlog+np.log(n)+np.log(ac)+np.log(s1)*k+
            #               np.log(s2)*(n-k)+np.log(S1)*(l1-1)+
            #               np.log(S2)*(l2-1)+np.log(((S1-a*l1)*(S2-a*l2)-l1*l2*ac**2)))-
            #              np.log(k*(n-k)*(a**2+ac**2)+a*ac*(k**2+(n-k)**2)))
            #     probn_log += addlog
            #import pdb;pdb.set_trace()
        probn = probn/(2*Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))#(1-(2*Ns-1)*a + 2*Ns**2*(Ns-1)*(a**2-ac**2))
        #probn_log = probn_log/(2*Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))#(1-(2*Ns-1)*a + 2*Ns**2*(Ns-1)*(a**2-ac**2))
        res.append(probn)
    return res


from scipy.special import gammaln

def log2comb(n, k):
    return (gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)) 

def avs_distmod2ovl_log(Ns,a,ac):
    res = []
    for n in range(1,2*Ns+1):
        probn = 0
        for k in range(n+1):
            s1,s2,l1,l2 = k*a+(n-k)*ac,(n-k)*a+k*ac,Ns-k,Ns-(n-k)
            S1,S2 = 1-s1,1-s2
            mult = log2comb(Ns,k)+log2comb(Ns,n-k)
            if mult >= 0:
               add = (mult+np.log(n)+np.log(ac)+np.log(s1)*k+np.log(s2)*(n-k)+np.log(S1)*(l1-1)+np.log(S2)*(l2-1)+np.log(((S1-a*l1)*(S2-a*l2)-l1*l2*ac**2)))- np.log(k*(n-k)*(a**2+ac**2)+a*ac*(k**2+(n-k)**2))
               #import pdb;pdb.set_trace()       
               probn+=np.exp(add)
        #import pdb; pdb.set_trace()
        probn = probn/(2*Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))#(1-(2*Ns-1)*a + 2*Ns**2*(Ns-1)*(a**2-ac**2))
        res.append(probn)
    return res

# x = np.linalg.lstsq(np.array([[1,np.log(a)] for a in np.arange(10,70)]),np.log(list(dist.values())[10:70]))[0]
# plt.loglog(np.arange(10,70),list(dist.values())[10:70])
# plt.loglog(np.arange(10,70),np.exp([x[1]*np.log(a) + x[0] for a in np.arange(10,70)]),
#            label=r'ols fit $N_e \sim \alpha^{'+str(x[1])+'}$')
# plt.legend()

# #%matplotlib qt 
# plt.imshow(np.array([[np.log(x[1]) if x[1] > 0 else 0 for x in xr] for xr in exp_dists]),origin='lower')
# plt.colorbar()
# #plt.plot([exp_dists[i,80-i][0] for i in range(100)])
# #plt.plot([100*exp_dists[i,80-i][1] for i in range(100)])
# #45,45 should be critical

def induced_avs_distmod2ovl(Ns,a,ac):
    """important, also k=0 computed..."""
    res = []
    for k in range(0,Ns+1):
        probn = 0
        for j in range(1,Ns+1):
            s1,s2,l1,l2 = k*a+(j)*ac,(j)*a+k*ac,Ns-k,Ns-(j)
            S1,S2 = 1-s1,1-s2
            mult = binom(Ns,k)*binom(Ns,j)
            if mult >= 0:
               add = mult*j*ac*s1**k*s2**(j)*S1**(l1-1)*S2**(l2-1)*((S1-(a*l1))*(S2-(a*l2))-(l1*l2)*ac**2) / \
                      (k*(j)*(a**2+ac**2)+a*ac*(k**2+(j)**2))
               probn+=add 
        # here not divided by 2 since probability avs starts in s2 is 0.5
        probn = probn/(Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))
        res.append(probn)
    return np.array(res)


from scipy.special import gammaln

def log2comb(n, k):
    return (gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)) 

def induced_avs_distmod2ovl_log(Ns,a,ac):
    res = []
    for k in range(0,Ns+1):
        probn = 0
        for j in range(1,Ns+1):
            s1,s2,l1,l2 = k*a+(j)*ac,(j)*a+k*ac,Ns-k,Ns-(j)
            S1,S2 = 1-s1,1-s2
            mult = log2comb(Ns,k)+log2comb(Ns,j)
            if mult >= 0:
               add = (mult+np.log(j)+np.log(ac)+np.log(s1)*k+np.log(s2)*(j)+np.log(S1)*(l1-1)+np.log(S2)*(l2-1)+np.log((S1-(a*l1))*(S2-(a*l2))-(l1*l2)*ac**2)) - \
                      np.log((k*(j)*(a**2+ac**2)+a*ac*(k**2+(j)**2)))
               probn+=np.exp(add)
        # here not divided by 2 since probability avs starts in s2 is 0.5
        probn = probn/(Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))
        res.append(probn)
    return np.array(res)


def started_avs_distmod2ovl(Ns,a,ac):
    """important, also k=0 computed..."""
    res = [0]
    for j in range(1,Ns+1):
        probn = 0
        for k in range(0,Ns+1):
            s1,s2,l1,l2 = k*a+(j)*ac,(j)*a+k*ac,Ns-k,Ns-(j)
            S1,S2 = 1-s1,1-s2
            mult = binom(Ns,k)*binom(Ns,j)
            if mult >= 0:
               add = mult*j*ac*s1**k*s2**(j)*S1**(l1-1)*S2**(l2-1)*((S1-(a*l1))*(S2-(a*l2))-(l1*l2)*ac**2) / \
                      (k*(j)*(a**2+ac**2)+a*ac*(k**2+(j)**2))
               probn+=add 
        # here not divided by 2 since probability avs starts in s2 is 0.5
        probn = probn/(Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))
        res.append(probn)
    return np.array(res)


from scipy.special import gammaln

def log2comb(n, k):
    return (gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)) 

def started_avs_distmod2ovl_log(Ns,a,ac):
    res = [0]
    for j in range(1,Ns+1):
        probn = 0
        for k in range(0,Ns+1):
            s1,s2,l1,l2 = k*a+(j)*ac,(j)*a+k*ac,Ns-k,Ns-(j)
            S1,S2 = 1-s1,1-s2
            mult = log2comb(Ns,k)+log2comb(Ns,j)
            if mult >= 0:
               add = (mult+np.log(j)+np.log(ac)+np.log(s1)*k+np.log(s2)*(j)+np.log(S1)*(l1-1)+np.log(S2)*(l2-1)+np.log((S1-(a*l1))*(S2-(a*l2))-(l1*l2)*ac**2)) - \
                      np.log((k*(j)*(a**2+ac**2)+a*ac*(k**2+(j)**2)))
               probn+=np.exp(add)
        # here not divided by 2 since probability avs starts in s2 is 0.5
        probn = probn/(Ns*(1-a*(2*Ns-1)+(a**2-ac**2)*(Ns*(Ns-1))))
        res.append(probn)
    return np.array(res)


def sum_dist(started,induced):
    res = np.zeros(len(started)+len(induced)-1)
    for z in range(len(res)):
        for k in range(z+1):
            if (0<=k<len(started)) and (0<=z-k<len(induced)):
                #print('comb ',z,k,z-k)
                res[z] += started[k]*induced[z-k]
    return res

# N_single = 50
# N = 2*N_single
# a = np.linspace(0,1/N_single,100)[87];ac = np.linspace(0,1/N_single,100)[1]
# dist = dists[87,1]
# detailed_avs = [[avs_dist_nk(N_single,a,ac,k,n-k) for k in range(n+1)] for n in range(1,N+1)]
# joint_dist = [[avs_dist_nk(N_single,a,ac,ns1,ns2) for ns1 in range(N_single+1)]
#                for ns2 in range(N_single+1)]
# sd = [np.sum([np.sum(x) for x in xr]) for xr in detailed_avs]
# assert(np.sum(np.abs(sd-dist)) < 1e-10)
# detailed_array = np.array([[np.sum(x) for x in xr]+[0]*(len(detailed_avs[-1])-len(xr)) for xr in detailed_avs])

# check finite size scaling
# #%matplotlib inline
# plt.figure(figsize=(6,4))
# N_singles = [25,50,100,250,500]
# a_idx = 1
# ac_idx = 200-a_idx
# for N_single in N_singles:
#     N = 2*N_single
#     acrit = ehe_critical_weights(N)[0,0]
#     alpha = acrit*a_idx/100
#     alpha_cross = acrit*ac_idx/100
#     dist = avs_distmod2ovl_log(N_single,alpha,alpha_cross)
#     dist_rescaled = [N**(1.5)*d for d in dist]
#     x_rescaled = np.arange(1,len(dist)+1)/N
#     plt.loglog(x_rescaled,dist_rescaled,label=str(N))
# N_single = 50
# plt.legend()
# plt.title(r'$\alpha=\alpha_{crit}/3$')
# plt.ylabel(r'$N^{\frac{3}{2}}P$')
# plt.xlabel(r'avs/N')

# %matplotlib qt
# plt.plot([1,2,3])

def cross_talk_p(N_single,alpha,alpha_cross,s_0,norm=True,comp_size=None):
    if comp_size is not None:
        detailed_avs = [avs_dist_nk(N_single,alpha,alpha_cross,comp_size-k,k)[0] for k in range(N_single+1)]
        return (np.sum(detailed_avs[s_0:])/np.sum(detailed_avs),detailed_avs)
    dist = avs_distmod2ovl_log(N_single,alpha,alpha_cross)
    joint_dist = [[avs_dist_nk(N_single,alpha,alpha_cross,ns1,ns2) for ns1 in range(N_single+1)]
                for ns2 in range(N_single+1)]
    dist_subnet_1 = np.array([[x[0] for x in xr] for xr in joint_dist])
    inds = list(range(s_0,N_single+1))
    ret = np.sum(dist_subnet_1[np.ix_(inds,inds)])*2
    if norm:
        #import pdb;pdb.set_trace()
        ret = ret / np.sum(dist[int(2*s_0):])
    return ret
# calculate probability that avs started in subnet1 crosses over to subnet2 and has at least s_0
# units in both subnetworks

#joint_0 = np.array([[np.sum(x) for x in xr] for xr in joint_dist])
#subnet_avs = np.sum(joint_0,axis=0)

def cross_talk_p2(N_single,alpha,alpha_cross,s_0,comp_size):
    dist = avs_distmod2ovl_log(N_single,alpha,alpha_cross)
    detailed_avs = [[avs_dist_nk(N_single,alpha,alpha_cross,n-k,k)[0] for k in range(s_0,N_single+1)] for n in range(comp_size,2*N_single+1)]
    ret = np.sum([np.sum(detailed_avs)])
    return ret
# calculate probability that avs started in subnet1 crosses over to subnet2 and has at least s_0
# units in both subnetworks

#joint_0 = np.array([[np.sum(x) for x in xr] for xr in joint_dist])
#subnet_avs = np.sum(joint_0,axis=0)

from itertools import chain, combinations
import sympy as sym

def W_ring(n,alpha,k=1,per_bc = True,self_activation=True,dtype=float):
    W = np.zeros((n,n),dtype=dtype)
    for i in range(n):
        #if i-1>=0:
        for j in range(1,k+1): 
            W[i,(i-j if i-j >= 0 else i-j+n)] = alpha
        if self_activation:    
            W[i,i] = alpha
        #if i+1<n:
        for j in range(1,k+1):
            W[i,(i+j if i+j<n else i+j-n)] = alpha
    #if per_bc:
    #    W[0,n-1] = alpha
    #    W[n-1,0] = alpha
    return W


class RingModelAvalancheSizeDistributionDet():
    def __init__(self,N,alpha,k=1,deltaU=0.022,self_activation=True,symbolic=False):
        self.alpha = alpha
        self.N = N
        self.k = k
        self.W = W_ring(N,alpha,k=k,per_bc=True,self_activation=self_activation,dtype=object if symbolic else float)
        self.deltaU = deltaU
        self.volumes = {}
        self.detailed_volumes = {}
        self.probs = {}
        self.det_func = np.linalg.det \
                        if not symbolic else lambda a: sym.Matrix(a).det()


    def detailed_volume(self,i_s,n,holes,num_ass):
        vol = self.detailed_volumes.get((i_s,n,holes,num_ass))
        if vol is not None:
            return vol
        # Calculate volume of avalanche size with determinant formula
        det = self.det_func
        #J = set(range(n))
        J = [0]
        idx=0
        for h in holes:
            idx+=1+h
            J.append(idx)
        J = set(J)    
        Jc = set(range(N))-J#list(range(n,self.N))
        vol_is = self.deltaU
        vol_J = 1
        rem_u = list(J-set([i_s]))
        if len(rem_u) > 0:
            m = np.diag([np.sum(self.W[i,list(J)]) for i in rem_u]) \
                                 - self.W[np.ix_(rem_u,rem_u)]
            vol_J = det(m) 
            #rint('vol_J',vol_J)
        vol_Jc = 1
        #print('Jc',Jc)
        if len(Jc) > 0:
            m = np.diag([1 - np.sum(self.W[i,list(J)]) for i in list(Jc)]) \
                                    - self.W[np.ix_(list(Jc),list(Jc))]
            #print('mc',m)                        
            vol_Jc = det(m)
            #print('vol_Jc',vol_Jc)
        vol = vol_is*vol_J*vol_Jc
        self.detailed_volumes[(i_s,n,holes,num_ass)] = vol
        return vol

    def volume_size0(self):
        vol = 0
        for i in range(self.N):
            m = np.diag([1 if j!=i else 1-self.deltaU for j in range(self.N)]) -self.W
            vol = vol + self.det_func(m)
        return vol

    def noninh_volume(self):
        return self.det_func(np.eye(self.N,dtype=int)-self.W)


    def is_equivalent(self,hs,snh):
        #print('is_equiv',hs,list(reversed(hs)),snh,(hs in snh) or (tuple(reversed(hs)) in snh)) 
        return (hs in snh) or False#(tuple(reversed(hs)) in snh)

    def hole_sequences(self,n):
        all_sequences = list(itertools.product(*[(range(self.k))]*(n-1)))
        all_sequences.sort(key=np.sum)
        grouped = [(x[0],list(x[1])) for x in itertools.groupby(all_sequences,lambda x:np.sum(x))]
        #print('grouped',grouped)
        distinct_sequences = []
        for num_holes,sequences in grouped:
            snh = []
            #print('sequences with num_holes',num_holes,sequences)
            for hs in sequences:
                if not(self.is_equivalent(hs,snh)):
                    snh.append(hs)
            distinct_sequences.extend(snh)
        return distinct_sequences

    def num_assignments(self,J):
        return len(set([frozenset([(j+i)%self.N for j in J]) for i in range(self.N)]))

    def volume(self,n):
        vol = self.volumes.get(n)
        if vol is not None:
            return vol
        #print("n",n)    
        if n == 0:
            vol = self.volume_size0()
            self.volumes[n] = vol
            return vol
        vol = 0
        hole_sequence = self.hole_sequences(n)
        #print('n',n,'hole_seqs',hole_sequence)
        for holes in hole_sequence:
            if n+np.sum(holes)<self.N or np.sum(holes)==0:
                J = [0]
                idx=0
                for h in holes:
                    idx+=1+h
                    J.append(idx)
                num_ass = 1 if n+np.sum(holes)==self.N else self.num_assignments(J)#self.N# how many ways to select n units for this circle part
                for i_s in J:
                    vol += num_ass*self.detailed_volume(i_s,n,holes,num_ass)
            self.volumes[n] = vol
        return vol

    def prob(self,n):
        prob = self.probs.get(n)
        if prob is not None:
            return prob
        print('n',n)
        norm_n = self.volume(n)/self.N # 1/N prob to get activation to correct
        norm_0 = self.volume(0)/self.N
        norm_noninh = self.noninh_volume()
        prob = norm_n/(norm_noninh - norm_0)
        #prob = norm_n/norm_0
        self.probs[n] = prob
        return prob

    def avs_dist(self):
        for i in range(1,self.N+1):
            self.prob(i)
        return self.probs

# alpha=sym.symbols('a');
# N=5;
# U_start_end = [2*alpha] + [2*alpha]*(N-2) + [alpha]
# #vol1=sym.simplify(np.prod(U) - np.prod([U[i]-2*alpha*np.cos(2*np.pi*i/N) for i in range(N)]))
# W = W_ring(N,alpha,self_activation=False,dtype=object,per_bc=False);
# if N>= 2:
#     W[0,N-1] = 0;
#     W[N-1,0] = 0
# vol = sym.simplify((sym.Matrix(np.diag(U_start_end)-W).det()))
# assert(sym.simplify(vol - alpha**N) == 0)


# U_norm_line = [1]*(N)
# #vol1=sym.simplify(np.prod(U) - np.prod([U[i]-2*alpha*np.cos(2*np.pi*i/N) for i in range(N)]))
# W = W_ring(N,alpha,self_activation=False,dtype=object,per_bc=False);
# if N>= 2:
#     W[0,N-1] = 0;
#     W[N-1,0] = 0
# indices = list(range(N))
# index = 2
# del indices[index]
# del U_norm_line[index]
# W_norm_line = W[np.ix_(indices,indices)]
# vol = sym.simplify((sym.Matrix(np.diag(U_norm_line)-W_norm_line).det()))
# #assert(sym.simplify(vol - alpha**N) == 0)

# U_start_end_N = [2*alpha] + [2*alpha]*(N-2) + [2*alpha]
# #vol1=sym.simplify(np.prod(U) - np.prod([U[i]-2*alpha*np.cos(2*np.pi*i/N) for i in range(N)]))
# W = W_ring(N,alpha,self_activation=False,dtype=object,per_bc=False);
# if N>= 2:
#     W[0,N-1] = 0;
#     W[N-1,0] = 0
# vol = sym.simplify((sym.Matrix(np.diag(U_start_end_N)-W).det()))
# #assert(sym.simplify(vol - alpha**N) == 0)


# U_start_inside = [alpha] + [2*alpha]*(N-2) + [alpha]
# inside = 3
# #vol1=sym.simplify(np.prod(U) - np.prod([U[i]-2*alpha*np.cos(2*np.pi*i/N) for i in range(N)]))
# W = W_ring(N,alpha,self_activation=False,dtype=object,per_bc=False);
# if N>= 2:
#     W[0,N-1] = 0;
#     W[N-1,0] = 0

# indices = list(range(N))
# index = 2
# del indices[index]
# del U_start_inside[index]
# W_inside = W[np.ix_(indices,indices)]
# W_inside =np.diag(U_start_inside)-W_inside
# vol = sym.simplify((sym.Matrix(W_inside).det()))
# assert(sym.simplify(vol - alpha**(N-1) == 0))


# # Jetzt fehlen  nur noch upper boundaries -> das ist extrem cool

# alpha = 1/4

# W = W_ring(N,alpha,self_activation=False,dtype=object,per_bc=False);
# if N>= 2:
#     W[0,N-1] = 0;
#     W[N-1,0] = 0

# U_rem = [1-alpha]+[1]*(N-2) + [1-alpha]
# W_rem = np.diag(U_rem)-W
# vol = sym.simplify((sym.Matrix(W_rem).det()))

# lambda1 = 1/2 + np.sqrt(1/4 - alpha**2)
# lambda2 = 1/2 - np.sqrt(1/4 - alpha**2)

# N1 = N-1
# C = (1-alpha-lambda2)/(lambda1-lambda2)
# D = 1-C
# vt_n = lambda N1: C*lambda1**N1 + D*lambda2**N1
# v = sym.simplify(C*lambda1**N1 + D*lambda2**N1)

# vrec_n = lambda N:(1-alpha)*vt_n(N-1)- alpha**2*vt_n(N-2)
# s = sym.simplify(vrec_n(N)-vol)


# # normalization constant calculation
# W_norm = np.eye(N) - W
# Cr = (1-lambda2)/(lambda1-lambda2)
# Dr = 1-Cr
# rn = lambda N:Cr*lambda1**N+Dr*lambda2**N
# r = rn(N)
# vol = sym.simplify((sym.Matrix(W_norm).det()))
# abs(vol - r)

def v_n(n,alpha):
    if n == 0:
        return 1
    if n == 1:
        return 1-2*alpha
    return vrec_n(n,alpha)

# now calculate analytical closed form solution
def avs_ring(N,alpha):
    l1,l2 = 1/2 + (1/4 - alpha**2)**(1/2),1/2 - (1/4 - alpha**2)**(1/2)
    C = (1-alpha-l2)/(l1-l2)
    D = 1-C
    C_norm = (1-l2)/(l1-l2)
    D_norm = 1-C_norm
    norm = C_norm*l1**(N-1)+D_norm*l2**(N-1)
    def v_rec(n):
        if n == 1:
            return (1-2*alpha)
        if n == 2:
            return (1-alpha)**2 - alpha**2
        fn1 = C*l1**(n-1)+D*l2**(n-1)
        fn2 = C*l1**(n-2)+D*l2**(n-2)
        return (1-alpha)*fn1-alpha**2*fn2
    def v(n):
        if n == 0:
            return 1
        if n == 1:
            return 1-2*alpha
        return v_rec(n)
    ret = np.zeros((N,))
    for n in range(1,N+1):
        ret[n-1] = n*alpha**(n-1)*v(N-n)#/norm
    return (ret/norm,norm)
# all bugs in avs_ring fixed now

def avs_line(N,alpha):
    l1,l2 = 1/2 + (1/4 - alpha**2)**(1/2),1/2 - (1/4 - alpha**2)**(1/2)
    C = (1-alpha-l2)/(l1-l2)
    D = 1-C
    C_norm = (1-l2)/(l1-l2)
    D_norm = 1-C_norm
    norm_r = lambda N:C_norm*l1**(N)+D_norm*l2**(N)
    norm = np.sum([norm_r(j)*norm_r(N-j-1) for j in range(N)])/N  
    def v_rec(n):
        if n == 1:
            return (1-alpha)
        # if n == 2:
        #     return (1-alpha) - alpha**2
        # fn1 = C*l1**(n-1)+D*l2**(n-1)
        # fn2 = C*l1**(n-2)+D*l2**(n-2)
        # return (1-alpha)*fn1-alpha**2*fn2
        return C*l1**(n)+D*l2**(n)

    def v(n):
        if n == 0:
            return 1
        if n == 1:
            return 1-alpha
        return v_rec(n)
    ret = np.zeros((N,))
    for n in range(1,N+1):
        ret[n-1] = 1/N*n*alpha**(n-1)*np.sum([v(j)*v(N-n-j) for j in range(N-n+1)])#/norm
    return (ret/norm,norm)
# for avs_line we have disconnected thingies for the normalization constant calculation
# -> das ist fixed...
# bei v_rec steckt noch der Wurm drin

def ehe_branching_factor(alpha,N):
    dist = []
    for n in range(1,N+1):
        r = (alpha/N)**n*(1-alpha/N)**(N-n-2)*(1-(alpha/N)*(N-n))
        if n==N-1:
            r = (alpha/N)**n
        dist.append(r)
    prob = np.array(dist)/(np.sum(dist))
    ret = 0
    for n,p in zip(range(1,len(prob)+1),prob):
        ret += n*p
    return ret
        
 # TODO Compute these numerically -> after abstract

def ehe_entropy(alpha,N):
    ehe_ana = [(1/N)*binom(N,n)*(n*alpha/N)**(n-1)*(1-(n*alpha/N))**(N-n-1)*((1-alpha)/(1-((N-1)/N)*alpha))  for n in range(1,N+1)]
    return -np.nansum([p*np.log2(p) for p in ehe_ana])


def ea(alpha,N):
    ehe_ana = [(1/N)*binom(N,n)*(n*alpha/N)**(n-1)*(1-(n*alpha/N))**(N-n-1)*((1-alpha)/(1-((N-1)/N)*alpha))  for n in range(1,N+1)]
    return np.array(ehe_ana)

from itertools import chain, combinations
import sympy as sym

def W_nn(N,alpha,dtype=float):
    W = np.zeros((N**2,N**2),dtype=dtype)
    unit_coords = [(i,j) for i in range(N) for j in range(N)]
    dmodn = lambda a,b,n: min((a-b)%N,(b-a)%N)
    ham_dist_modN = np.array([[dmodn(x1,x2,N)+dmodn(y1,y2,N) for (x2,y2) in unit_coords] for (x1,y1) in unit_coords])
    # dist_coords = lambda p1,p2:dmodn(p1[0],p2[0])+dmodn(p1[1],p2[1])
    # W = np.array([[dist_coords(unit_coords[  for j in range(N)] for i in range(N)]
    W[ham_dist_modN <= 1] = alpha
    return W


def W_ass(ass):
    """assigmnents=[1,4,6,2] means 1 unit with 1 conn.units. 4 with 2 3 with 3 2 with 4"""
    W = np.zeros((np.sum(ass),np.sum(ass)))
    L_add = np.ones(np.sum(ass))
    L_current = np.ones(np.sum(ass))
    #L_goal = [[l]**ass[l+1] for 
    unit = 0;
    for num_conn in range(len(ass),0,-1):
        num_ass = ass[num_conn-1];
        for a_idx in range(num_ass):
            #L[unit] = 
            unit += 1

class ModSimple2ovlAvalancheSizeDistributionDet():
    def __init__(self,Ns1,Ns2,alpha,alpha_cross,deltaU=0.022,symbolic=False):
        self.Ns1 = Ns1
        self.Ns2 = Ns2
        self.alpha_cross = alpha_cross
        self.alpha = alpha
        self.N = self.Ns1+self.Ns2
        self.W = W_mod2ovl(Ns1,Ns2,alpha,alpha_cross,dtype=object if symbolic else float)
        self.deltaU = deltaU
        self.volumes = {}
        self.detailed_volumes = {}
        self.probs = {}
        self.det_func = np.linalg.det \
                        if not symbolic else lambda a: sym.Matrix(a).det()


    def detailed_volume(self,i_s,J):
        J = frozenset(J)
        #print('i_s','J',i_s,J)
        vol = self.detailed_volumes.get((i_s,J))
        if vol is not None:
            return vol
        # Calculate volume of avalanche size with determinant formula
        det = self.det_func
        rem_u = list(J-set([i_s]))
        Jc = list(set(range(self.N))-J)
        vol_is = self.deltaU
        vol_J = 1
        if len(rem_u) > 0:
            m = np.diag([np.sum(self.W[i,list(J)]) for i in rem_u]) \
                                 - self.W[np.ix_(rem_u,rem_u)]
            #print('m',m)
            vol_J = det(m) 
            #print('vol_J',vol_J)
        vol_Jc = 1
        if len(Jc) > 0:
            m = np.diag([1 - np.sum(self.W[i,list(J)]) for i in Jc]) \
                                    - self.W[np.ix_(Jc,Jc)]
            #print('mc',m)                        
            vol_Jc = det(m)
            #print('vol_Jc',vol_Jc)
        vol = vol_is*vol_J*vol_Jc
        self.detailed_volumes[(i_s,J)] = vol
        return vol


    def volume_size0(self):
        vol = 0
        for i in range(self.N):
            m = np.diag([1 if j!=i else 1-self.deltaU for j in range(self.N)]) -self.W
            vol = vol + self.det_func(m)
        return vol

    def noninh_volume(self):
        return self.det_func(np.eye(self.N,dtype=int)-self.W)

    def volume(self,n):
        vol = self.volumes.get(n)
        if vol is not None:
            return vol
        #print("n",n)    
        if n == -1:
            vol = self.noninh_volume()
            self.volumes[n] = vol
            return vol
        if n == 0:
            vol = self.volume_size0()
            self.volumes[n] = vol
            return vol
        vol = 0
        for ns1 in range(min(self.Ns1+1,n+1)):
            ns2 = n - ns1
            if ns2 <= self.Ns2:
                   num_ass = binom(self.Ns1,ns1)*binom(self.Ns2,ns2) 
                   units = list(range(ns1))+list(range(self.Ns1,self.Ns1+ns2))
                   for i_s in units:
                        vol += num_ass*self.detailed_volume(i_s,set(units))
        self.volumes[n] = vol
        return vol

    def prob(self,n):
        prob = self.probs.get(n)
        if prob is not None:
            return prob
        norm_n = self.volume(n)/self.N # 1/N prob to get activation to correct
        norm_0 = self.volume(0)/self.N
        norm_noninh = self.volume(-1)

        prob = norm_n/(norm_noninh - norm_0)
        self.probs[n] = prob
        return prob

    def avs_dist(self):
        for i in range(1,self.N+1):
            self.prob(i)
        return self.probs

# enumerate possible graphical configurations
origin = (0,0)

def neighbors(pos):
    x,y = pos
    return [(x+dx,y+dy) for dx in [-1,0,1] for dy in [-1,0,1] if not ((dx==0 and dy==0) or (dx!=0 and dy!=0))]


def attachment_points(conf,max_n):
    atp = []
    lb,ub = np.min([c[0] for c in conf]),np.min([c[1] for c in conf])
    rb,db = np.max([c[0] for c in conf]),np.max([c[1] for c in conf])
    for pos in conf:
        for nb in neighbors(pos):
            if nb not in conf:
                if (max(rb,nb[0]) - min(nb[0],lb) < max_n) and (max(db,nb[1]) - min(nb[1],ub) < max_n):
                    atp.append(nb)
    return atp


def translate_to_origin(conf):
    lb,up = np.min([c[0] for c in conf]),np.min([c[1] for c in conf])
    return [(cx-lb,cy-up) for (cx,cy) in conf]

def rotate_90_deg(conf):
    R = np.array([[0,1],[-1,0]],dtype=int)
    return [R@c for c in conf]

def mirror_x(conf):
    return translate_to_origin([[-cx,cy] for cx,cy in conf])

def mirror_y(conf):
    return translate_to_origin([[cx,-cy] for cx,cy in conf])

def is_equiv(conf,confs):
    #print('confs_to_compare',confs)
    for conf in [set(translate_to_origin(conf)),set(mirror_x(conf)),set(mirror_y(conf))]:
        # print('conf',conf)
        if conf in confs:
            return True
        for _ in range(3):
           conf = set(translate_to_origin(rotate_90_deg(conf)))
           # print('rotated',conf)
           if conf in confs:
               return True
    return False


def generate_configurations(n,conf_p,max_n):
    """conf_p dict of [0,...n-1] configurations. Each is a list of coordinates"""
    if n==1:
        return {1:[set([origin])]}
    conf_n= []
    for conf in conf_p[n-1]:
        # print('conf ', conf)
        atp = attachment_points(conf,max_n)
        for ap in atp:
            # print('attachment_point',ap)
            conf_c = set(translate_to_origin(conf.union([ap])))
            # print('conf_c',conf_c)
            if not is_equiv(conf_c,conf_n):
                conf_n.append(conf_c)
    conf_p[n]=conf_n            
    return conf_p


    

def vis(conf):
    conf = translate_to_origin(conf)
    n,m = np.max([c[1] for c in conf])+1,np.max([c[0] for c in conf])+1
    W = np.zeros((n,m))
    for cx,cy in conf:
        W[cy,cx] = 1
    return W

def gen_configs_up_to(n):
    conf_p = {}
    for i in range(1,n**2+1):
        print(i)
        conf_p = generate_configurations(i,conf_p,n)
    return conf_p

def torus2inhabited(units,W):
    """transforms units from the N-torus to the corresponding point in the inhabited region"""
    transformation = np.eye(W.shape[0],dtype=int)-W
    offset = np.sum(W,axis=1)
    trans_units = (transformation @ units) + offset
    trans_temp = trans_units.copy()
    handle_avalanche(None,trans_units,W,U=1)
    return trans_units,trans_temp

def points2pdf(points,lower_limit=1,upper_limit=None,return_unique=True):
    if upper_limit is None:
        upper_limit = max(points)
    unique,counts = np.unique(points,return_counts=True)
    pdf = np.zeros(upper_limit - lower_limit + 1)
    for u,c in zip(unique,counts):
        pdf[u - lower_limit] = c
    if return_unique:
        return (pdf/np.sum(pdf),np.array(list(range(lower_limit,upper_limit+1))))
    return pdf/np.sum(pdf)

import matplotlib.pyplot as plt

def loglogplot(pdf,ax = None, figsize=(6,4),from_pdf=False,label=None):
    if from_pdf:
        norm_counts,unique = pdf
    else:
        points = pdf
        unique,counts = np.unique(points,return_counts=True)
        norm_counts = counts/np.sum(counts)
    if ax is None:
        f,ax = plt.subplots(figsize=figsize)
    ax.loglog(unique,norm_counts,label=label)
    #ax.set_xscale('log',nonposx='mask');
    #ax.set_yscale('log',nonposy='mask');
    #ax.scatter(unique,norm_counts,facecolors='none',edgecolors='b')
    return ax

def storekey(res,key):
   def inner(value):
       res[key] = value
   return inner
