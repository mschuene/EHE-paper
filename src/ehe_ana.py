
from itertools import chain, combinations
import numpy as np

def spiking_patterns(N):
    """a avalanche pattern is expressed in the format [(i,),(i_2_1,i_2,2...),...,(...,i_D_k)]"""
    for avalanche_size in range(N,0,-1):
        #print("avalanche_size ", avalanche_size)
        if avalanche_size == 1:
            for unit in range(N):
                yield [(unit,)]
        else:
            for length in range(avalanche_size,1,-1):
                #print("combinations",range(N),length)
                for units in combinations(range(N),avalanche_size):
                    #print("units ",units)
                    yield from sp_length_units(length,units)

def group_sizes(N,length):
    """list possible sizes of groups where the first group has size 1 and the sum of all groups has to equal length.
       can be reduced to choosing length-2 spaces between the remaining N-1 elements"""
    if length == 2:
        yield [1,N-1]
    else:
        for sel_spaces in combinations(range(2,N),length-2):
            yield np.diff((0,1)+sel_spaces+(N,))

def unit_placements(units,gs):
    if len(gs) == 0:
        yield []
    else: 
        for sel_units in combinations(units,gs[0]):
            for rest_placement in unit_placements(units-set(sel_units),gs[1:]):
                yield [sel_units] + rest_placement

def sp_length_units(length,units):
    for gs in group_sizes(len(units),length):
        yield from unit_placements(set(units),gs)

def completing_patterns(pattern,N,set_of_patterns=None):
    if set_of_patterns==None:
        set_of_patterns = spiking_patterns(N)
    def is_prefix(pattern1,pattern2):
        fp1= np.array(reduce(lambda a,b:a+b,pattern1))
        fp2 = np.array(reduce(lambda a,b:a+b,pattern2))
        if len(fp2) <= len(fp1):
            return False
        return (fp1 == fp2[:len(fp1)]).all()
    return [p for p in set_of_patterns if is_prefix(pattern,p)]

class SpikingPatternVolumes():
    def __init__(self,W,deltaU=0.022):
        self.N = W.shape[0]
        self.sps= list(spiking_patterns(self.N))
        self.volumes = {tuple(sp):None for sp in self.sps}
        self.W = W
        self.normalized_volumes={}
        self.NR = NoninhabitedRegion(W)
        self.deltaU = deltaU

    def get_probs(self):
        vol = 0
        for sp in self.sps:
            vol +=self.volume(sp)
        self.normalized_volumes = {k:v/vol for k,v in self.volumes.items()}
        return self.normalized_volumes

    def volume(self,pattern):
        vol = self.volumes[tuple(pattern)]
        if vol is not None:
            return vol
        # now calculate volume... oh my
        N = self.N;W = self.W;deltaU=self.deltaU
        i = pattern[0][0]
        ip = set(reduce(lambda a,b:a+b,pattern))
        ipc = set(range(self.N)) - ip
        area_along_i = deltaU
        vol = area_along_i
        if len(ipc) > 0:
            upper_bounds = [1-sum(W[ic,i] for i in ip) for ic in ipc]
            #nr = sym.symbols('NONINH_'+'_'.join(str(i) for i in ipc)+"__"+"_".join(str(ub).replace(" ","_") for ub in upper_bounds))
            area_along_ipc = prod(upper_bounds) - self.NR.noninhabited_region(ipc,upper_bounds)
            vol*=area_along_ipc
        if len(pattern) > 1:
            area_along_p = prod(prod(sum(W[j,k] for k in pattern[d-1]) for j in pattern[d]) for d in range(1,len(pattern)))
            vol *= area_along_p
        self.volumes[tuple(pattern)] = vol
        return vol

def empirical_sp_dist(emp_spiking_patterns):
    unique,counts = np.unique(emp_spiking_patterns,return_counts=True)
    norm_count = np.sum(counts)
    return {tuple(tuple(s for s in step) for step in sp):c/norm_count for sp,c in zip(unique,counts)}

def avalanche_size_statistics(N,sp_probs):
    # group all avalanche patterns with the same avalanche size
    avs = [0 for i in range(N)]
    for sp,prob in sp_probs.items():
        size = sum(len(step) for step in sp)
        #print(sp,size)
        avs[size-1] += prob
    return avs

import sympy as sym
deltaU = 0.022
def overlap_av_vol(av_ovl,alpha,N,N_sub,deltaU=deltaU):
    D = len(av_ovl)
    ns1 = [step[0] for step in av_ovl]
    ns2 = [step[1] for step in av_ovl]
    no  = [step[2] for step in av_ovl]
    n_s1,n_s2,n_o = sum(ns1),sum(ns2),sum(no)
    N_s1,N_s2,N_o = N
    vol_av = deltaU*prod((alpha/N_sub)**(ns1[i]+ns2[i]+no[i])*(ns2[i-1]+no[i-1])**ns1[i]*
                      (ns2[i-1]+no[i-1])**ns2[i]*(ns1[i-1]+ns2[i-1]+no[i-1])**no[i]for i in range(1,D))
    vol_avc = ((1-(alpha/N_sub)(n_s1+n_o))**(N_s1-n_s1)*
               (1-(alpha/N_sub)(n_s2+n_o))**(N_s2-n_s2)*
               (1-(alpha/N_sub)(n_s1+n_s2+n_o))**(N_o-n_o)
               -(1-(alpha/N_sub)(n_s1+n_o))**(N_s1-n_s1-1)*
                (1-(alpha/N_sub)(n_s2+n_o))**(N_s2-n_s2-1)*
                (1-(alpha/N_sub)(n_s1+n_s2+n_o))**(N_o-n_o-1)
                *((alpha/N_sub)*(sum(N)-N_s1+N_s2+N_o)+
                  (alpha/N_sub)**2*(N_s1-n_s1)*(N_s2-n_s2)-
                  (alpha/N_sub)**3*(N_s1-n_s1)*(N_s2-n_s2)*(N_o-n_o))
                -(alpha/N_sub)*((N_s1-n_s1)*(1-(alpha/N_sub)*(n_s2+n_o))*(1-(alpha/N_sub)*(n_s1+n_s2+n_o))+
                                (N_s2-n_s2)*(1-(alpha/N_sub)*(n_s1+n_o))*(1-(alpha/N_sub)*(n_s1+n_s2+n_o))+
                                (N_o-n_o)*(1-(alpha/N_sub)*(n_s1+n_o))*(1-(alpha/N_sub)*(n_s2+n_o))))
    vol_avc = 1
    return vol_av*vol_avc

def homogen_av_vol(av_hom,alpha,N,deltaU=deltaU):
    D = len(av_hom)
    n = sum(av_hom)
    vol_av = deltaU*prod((alpha/N)**av_hom[i]*av_hom[i-1] for i in range(1,D))
    return vol_av

def av_hom_patterns(size):
    return {k:v for k,v in  zip(*np.unique([tuple([len(step) for step in sp]) for sp in spiking_patterns(size) ],return_counts=True))}

class S_bar():
    def __init__(self,N,alpha,k):
        self.s_bars = defaultdict(lambda:None);
        self.N = N
        self.alpha = alpha
        self.k = k

    def s_bar(self,m,l,j):
        m,l,j = tuple(m),tuple(l),tuple(j)
        res = self.s_bars[(m,l,j)]
        if res is not None:
            return res
        # now calculate s_bar by recursion formula
        js1,js2,jo = j
        ms1,ms2,mo = m
        ls1,ls2,lo = l
        ks1,ks2,ko = self.k
        N,alpha = self.N,self.alpha
        if sum(j) == 0:
            ns1,ns2,no = ks1+ls1,ks2+ls2,ko+lo
            res = ((1-(alpha/N)*(ns1+no))**ms1*(1-(alpha/N)*(ns2+no))**ms2*(1-(alpha/N)*(ns1+ns2+no))**mo -
                   (1-(alpha/N)*(ns1+no))**(ms1-1)*(1-(alpha/N)*(ns2+no))**(ms2-1)*(1-(alpha/N)*(ns1+ns2+no))**(mo-1)
                   *((alpha/N)*(ms1+ms2+mo)+(alpha/N)**2*ms2*ms2-(alpha/N)**3*ms1*ms2*mo)
                   -(alpha/N)*(ms1*(1-(alpha/N)*(ns2+no))*(1-(alpha/N)*(ns1+ns2+no))
                               +ms2*(1-(alpha/N)*(ns1+no))*(1-(alpha/N)*(ns1+ns2+no))
                               +mo*(1-(alpha/N)*(ns1+no))*(1-(alpha/N)*(ns2+no))))
        else:
            print([(is1,is2,io) for is1 in range(js1+1) for is2 in range(js2+1) for io in range(jo+1) if is1+is2+io > 0])
            res = sum(sym.binomial(ms1,is1)*sym.binomial(ms2,is2)*sym.binomial(mo,io)*(alpha/N)**(js1+js2+jo)
                      *(ls1+lo)**is1*(ls2+lo)**is2*(ls1+ls2+lo)**io
                      *self.s_bar((ms1-js1,ms2-js2,mo-jo),(is1,is2,io),(js1-is1,js2-is2,jo-io))
                      for is1 in range(js1+1) for is2 in range(js2+1) for io in range(jo+1) if is1+is2+io > 0)
        self.s_bars[(m,l,j)] = res
        #(ks1+ls1,ks2+ls2,ko+lo)
        return res


class orig_S_bar():
    def __init__(self,N,alpha,k):
        self.N = N;
        self.alpha = alpha
        self.k = k

    def s_bar(self,m,l,j):
        pass

alpha,N,deltaU = sym.symbols('alpha,N,deltaU')

size = 5

def factor_hom(s):
    avp = av_hom_patterns(s)
    print(set(avp))
    summed = sym.simplify(sum(homogen_av_vol(avh,alpha,N,deltaU) for avh in avp))
    print(summed)
    return summed/(deltaU*(alpha/N)**(s-1))
