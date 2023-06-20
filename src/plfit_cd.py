
import matplotlib.pyplot as plt
import numpy as np
import pickle

def f_pl(alpha,l,L):
    def f_not_1(x):
        range_mask = ((x >= l) & (x <= L))
        ret = np.zeros(x.shape)
        ret[range_mask] = ((alpha - 1)/(np.power(l,1-alpha) - 
                np.power(L,1-alpha)) * np.power(x[range_mask],-alpha))
        return ret
    def f1(x):
        range_mask = ((x >= l) & (x <= L))
        ret = np.zeros(x.shape)
        ret[range_mask] = 1/x[range_mask]*np.log(L/l)
        return ret
    return f_not_1 if alpha != 1 else f1

def S_pl(alpha,l,L):
    def S_not_1(x):
        range_mask = ((x >= l) & (x <= L))
        ret = np.zeros(x.shape)
        ret[range_mask] = ((np.power(x[range_mask],1-alpha)-np.power(L,1-alpha))/(np.power(l,1-alpha)-np.power(L,1-alpha)))
        ret[x < l] = 1
        return ret
    def S1(x):
        range_mask = ((x >= l) & (x <= L))
        ret = np.ones(x.shape)
        ret[range_mask] = (np.log(L) - np.log(x))/(np.log(L) -np.log(l))
        ret[x > L] = 0
        return ret
    return S_not_1 if alpha != 1 else S1

def draw_sample(l,L,alpha,N=10000):
      u = np.random.random(N)
      return l/np.power(1 - (1 - np.power(l/L,alpha-1))*u,1/(alpha-1))

from scipy.special import zeta

def hurwitz_zeta(alpha,l,L,return_cumsum=False):
    """l==np.Inf is the actual hurwitz zeta"""
    if (L-l) < 1e3 or return_cumsum:
        if L == np.Inf:
            L = 1e5
        return (np.cumsum(np.power(np.arange(l,L+1),-alpha)) if return_cumsum else
                np.sum(np.power(np.arange(l,L+1),-alpha)))
    else: # use scipy.special implementation which uses euler-mclaurin
        return zeta(alpha,l) - zeta(alpha,L) #zeta handles np.Inf


def discrete_f_pl(alpha,l,L):
    def d_f_pl(x):
        x = np.array(x)
        # x has to be vector of ints
        range_mask = ((x >= l) & (x <= L))
        ret = np.zeros(x.shape)
        norm_const = 1/hurwitz_zeta(alpha,l,L)
        ret[range_mask] = norm_const*np.power(x[range_mask],-alpha)
        return ret
    return d_f_pl

def discrete_S_pl(alpha,l,L):
    def d_S_pl(x):
        x = np.array(x)
        range_mask = ((x > l) & (x <= L)) # > important for cumsum index
        ret = np.ones(x.shape)
        hurwitz_cumsum = hurwitz_zeta(alpha,l,L,return_cumsum=True)
        zeta = hurwitz_cumsum[-1]
        hinds = (x[range_mask]-l-1).astype(int)
        ret[range_mask] = (zeta - hurwitz_cumsum[hinds])/zeta
        ret[x > L] = 0
        return ret;
    return d_S_pl

def draw_sample_discrete(l,L,alpha,N=1000):
    ret = np.zeros(N,dtype='i')
    beta = alpha-1
    umax = np.power(1/l,beta)
    umin = np.power(1/L,beta)
    f = discrete_f_pl(alpha,l,L)
    q = lambda y: np.power(l/y,beta) - np.power(l/(y+1),beta)
    #print(umax)
    for i in range(N):
        numit = 0
        while(numit<100):
            u = np.random.rand()*(umax-umin) + umin
            v = np.random.rand()
            y = np.floor(1/np.power(u,1/beta))
            tau = np.power(1 + 1/y,beta)
            b = np.power(l + 1,beta)
            #print(v*y*(tau - 1)/(b - np.power(l,beta)),(l*tau)/b)
            if v <= (np.power(y,-alpha)*q(l))/(np.power(l,-alpha)*q(y)):
#            if v*y*(tau - 1)/(b - np.power(l,beta)) <= (l*tau)/b:
                break
            numit += 1
        if(numit == 100):
            print('did not break')
        ret[i] = y
        # print(numit)
    return ret

def log_likelihood(l,L,x,dist_to_1=1e-6):
    r = l/L
    range_mask = ((x >= l) & (x <= L))
    samples_in_range = x[range_mask]
    log_g = np.sum(np.log(samples_in_range))/np.shape(samples_in_range)[0]
    def ll(alpha):
        if np.abs(alpha-1) > dist_to_1:
            return (np.log(alpha-1) - np.log(1 - np.power(r,alpha-1)) -
                    alpha*(log_g - np.log(l)) - np.log(l))
        else:
            return -np.log(-np.log(r)) - log_g
    return ll

def discrete_log_likelihood(l,L,x):
    samples_in_range = x[(x >= l) & (x <= L)]
    log_g = np.sum(np.log(samples_in_range))/np.shape(samples_in_range)[0]
    def ll(alpha):
        return - np.log(hurwitz_zeta(alpha,l,L)) - alpha * log_g
    return ll

from scipy.optimize import minimize_scalar
def ml_fit(sample,l,L,log_like=log_likelihood,bounds=[1,10]):
    log_like = log_like(l,L,sample)
    res = minimize_scalar(lambda x:-log_like(x),method='Bounded',bounds=bounds)
    assert(res.success)
    return res.x

def dist_KS(alpha,l,L,sample,null_ccdf=S_pl,return_ccdf=False):
    ccdf_pl = null_ccdf(alpha,l,L)
    sample = sample[((sample >= l) & (sample <= L))]
    print('start building empirical ccdf')
    unique,count = np.unique(sample,return_counts=True)
    sort_idx = np.argsort(unique)
    unique_s,count_s = (unique[sort_idx],count[sort_idx])
    cum_count = np.cumsum(count_s)
    S_e = np.ones(cum_count.shape[0])
    S_e[1:] = 1 - cum_count[1:]/cum_count[-1] # < und <= verwechselt kann sein
    print('build null ccdf')
    S = ccdf_pl(unique_s)
    print('calc max dev')
    max_dev = np.max(np.abs(S - S_e))
    if return_ccdf:
        return (max_dev,S,S_e,unique_s)
    else:
        return  max_dev

from scipy.optimize import minimize_scalar

def S_dist(f,l,L,idx0=1):
    """assumes that f is vector [x1,...] with xj being the prob for value j"""
    f_sub = f[(l-1):(L)]/np.sum(f[(l-1):(L)])
    ret = np.ones_like(f_sub)
    ret[1:] = 1-np.cumsum(f_sub)[:-1]
    return ret

def KS_pl_fit_dist(f,l,L,alpha_fix=None,bounds=[1,3]):
    S_f = S_dist(f,l,L)
    x = np.arange(l,L+1)
    print(S_f.shape,x.shape)
    if alpha_fix is None:
        res = minimize_scalar(lambda a:np.max(np.abs(discrete_S_pl(a,l,L,)(x) - S_f)),
                              method='Bounded',bounds=bounds)
        assert(res.success)
        return (res.x,res.fun)
    return np.max(np.abs(discrete_S_pl(alpha_fix,l,L)(x)-S_f))

def KS_dist(f1,f2,l,L):
    S_f1 = S_dist(f1,l,L)
    S_f2 = S_dist(f2,l,L)
    x = np.arange(l,L+1)
    return np.max(np.abs(S_f1-S_f2))

def KL_pl_fit_dist(f,l,L,alpha_fix=None,bounds=[1,3]):
    pdf = (f,np.arange(1,len(f)+1))
    if alpha_fix is None:
        res = minimize_scalar(lambda a:sym_kl(pdf,a,upper_limit=L,lower_limit=l),
                              method='Bounded',bounds=bounds)
        assert(res.success)
        return (res.x,res.fun)
    return (alpha_fix,sym_kl(pdf,alpha_fix,upper_limit=L,lower_limit=l))

def KS_p_value(d_e,alpha,l,L,N,sample_func=draw_sample,num_samples=1000,null_ccdf=S_pl,return_dists=False,use_dists=None):
    distances = (np.array([dist_KS(alpha,l,L,sample_func(l,L,alpha,N),null_ccdf=null_ccdf) 
                 for _ in range(num_samples)]) if use_dists is None else use_dists)
    print('distances_calculated')
    if not(return_dists):
        return len(distances[distances > d_e])/num_samples
    else:
        return (len(distances[distances > d_e])/num_samples,distances)

def p_value(d,distances):
    return len(distances[distances <= d])/len(distances)

def plfit_cd(sample,l,L,fix_exponent=None,use_dists=None,num_samples=1000,return_all=False,
             log_like=log_likelihood,bounds=[1,10],null_ccdf=S_pl,sample_func=draw_sample):
    exponent = (ml_fit(sample,l,L,log_like=log_like,bounds=bounds)
                if fix_exponent is None else fix_exponent)
    d_ks,S,S_e,unique_s = dist_KS(exponent,l,L,sample,null_ccdf=null_ccdf,return_ccdf=True)
    print('distks',d_ks)
    p_value = np.nan
    if num_smples > 0:
        p_value = KS_p_value(d_ks,exponent,l,L,len(sample),num_samples=num_samples,use_dists=use_dists,
                             sample_func=sample_func,null_ccdf=null_ccdf)
    return ((exponent,p_value) if not return_all else
            (exponent,d_ks,S,S_e,unique_s,p_value))


def plfit_cd_discrete(sample,l,L,fix_exponent=None,use_dists=None,num_samples=1000,return_all=False):
    return plfit_cd(sample,l,L,fix_exponent=fix_exponent, use_dists=use_dists,
                    num_samples=num_samples,return_all=return_all, log_like=discrete_log_likelihood,
                    bounds=[1,10],null_ccdf=discrete_S_pl,sample_func=draw_sample_discrete)

import os
def calc_dists(l,L,alpha,tid,outdir,distfile="../avalanches/distancesl5L1e3N10002.dat",
               sep=10,null_ccdf=S_pl,sample_func=draw_sample):
    # if not os.path.isfile(distfile):
    #     np.memmap(distfile, dtype='float64', mode='w+', shape=(1000))
    # distances = np.memmap(distfile,dtype='float64',mode='r+',shape=(1000))#np.load(distfile,mmap_mode='r+')
    print('start calculating distances')
    print('tid ',tid,sep*(tid-1),(sep*tid))
    dists = KS_p_value(0.2,3/2,5,1000,int(1e7),num_samples=sep,return_dists=True,sample_func=sample_func)[1]
    print('dists',dists)
    np.save(os.path.join(outdir,"dists"+str(tid)),dists)
