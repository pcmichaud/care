import numpy as np
import pandas as pd
from itertools import product
from functools import partial
from numba import njit, float64, int64, prange
from scipy.optimize import minimize
import warnings
warnings.simplefilter('ignore')

class param:
    def __init__(self,group, label, free = True, value = 0.0, stderr = np.nan, tvalue = np.nan):
        self.group = group
        self.label = label
        self.free = free
        self.value = value
        self.stderr = stderr
        self.tvalue = tvalue
        return


class survey:
    def __init__(self):
        self.n_scns = 4
        self.choice_set = [1,2,3]
        self.n_choices = len(self.choice_set)
        return
    def load(self,file = 'data/data_stated_preferences_sld.csv'):
        df = pd.read_csv(file,delimiter=';')
        df.set_index(['respid','scn'],inplace=True)
        self.data = df
        self.data['constant'] = 1
        self.n_resps = len(self.data.index.get_level_values(0).unique())
        self.n_tot = self.n_resps * self.n_scns
        print(self.data.describe().transpose())
        return
    def extract_attributes(self,attr=['liv_rpa', 'liv_ri', 'liv_chsld', 'sup_pri', 'sup_npf', 'svc_inf', 'svc_avq', 'svc_avd', 'cons', 'wait'], eps = 1):
        self.attr_labels = attr
        svc_inf = np.zeros((self.n_tot,self.n_choices))
        svc_avq = np.zeros((self.n_tot,self.n_choices))
        svc_avd = np.zeros((self.n_tot,self.n_choices))
        cons = np.zeros((self.n_tot,self.n_choices))
        wait = np.zeros((self.n_tot,self.n_choices))
        liv_rpa = np.zeros((self.n_tot,self.n_choices))
        liv_ri = np.zeros((self.n_tot,self.n_choices))
        liv_chsld = np.zeros((self.n_tot,self.n_choices))
        if 'sup_npf' in self.attr_labels:
            sup_pri = np.zeros((self.n_tot,self.n_choices))
        if 'sup_npf' in self.attr_labels:
            sup_npf = np.zeros((self.n_tot,self.n_choices))
        for i,c in enumerate(self.choice_set):
            svc_inf[:,i] = self.data['svc_inf'+str(c)]
            svc_inf[svc_inf[:,i]<1,i] = 1
            svc_inf[:,i] = np.log(svc_inf[:,i])
            svc_avq[:,i] = self.data['svc_avq'+str(c)]
            svc_avq[svc_avq[:,i]<1,i] = 1
            svc_avq[:,i] = np.log(svc_avq[:,i])
            svc_avd[:,i] = self.data['svc_avd'+str(c)]
            svc_avd[svc_avd[:,i]<1,i] = 1
            svc_avd[:,i] = np.log(svc_avd[:,i])
            cons[:,i] = (self.data['income'] - self.data['cost'+str(c)])/100
            cons[cons[:,i]<1,i] = 1
            cons[:,i] = np.log(cons[:,i])
            wait[:,i] = self.data['wait'+str(c)]
            liv_rpa[:,i] = np.where(self.data['liv'+str(c)]=='rpa',1,0)
            liv_ri[:,i] = np.where(self.data['liv'+str(c)]=='ri',1,0)
            liv_chsld[:,i] = np.where(self.data['liv'+str(c)]=='chsld',1,0)
            sup_pri[:,i] = np.where(self.data['sup'+str(c)]=='for-profit',1,0)
            sup_npf[:,i] = np.where(self.data['sup'+str(c)]=='non-profit',1,0)
        #return dict(zip(self.attr_labels,[liv_rpa, liv_ri, liv_chsld, svc_inf, svc_avq, svc_avd, cons]))
        return dict(zip(self.attr_labels,[liv_rpa, liv_ri, liv_chsld, sup_pri, sup_npf, svc_inf, svc_avq, svc_avd, cons, wait]))
    def extract_choices(self):
        cs = self.data['choice'].values-1
        return cs
    def create_dummies(self,var, values):
        map_labels = dict(zip(np.arange(len(values)),values))
        for i, v in enumerate(values):
            self.data[var+'_'+str(i)] = np.where(self.data[var]== v,1,0)
        print(var,map_labels)
        print(self.data[[var+'_'+str(i) for i in range(len(values))]].mean())
        return
    def extract_shifters(self, shifter_labels):
        shifters = self.data[shifter_labels].values
        return shifters

# Create a class for groups of parameters:
class params:
    def __init__(self):
        self.pars = []
        return
    def add_param(self, this):
        self.pars.append(this)
    def npars(self):
        return len(self.pars)
    def set_groups(self):
        self.groups = set([p.group for p in self.pars])
        return
    def ngroups(self):
        return len(self.groups)
    def nfreepars(self):
        return len([p for p in self.pars if p.free])
    def extract_freepars(self):
        return np.array([p.value for p in self.pars if p.free])
    def restrict(self,group,label,fix_value):
        for p in self.pars:
            if p.group == group and p.label == label:
                p.free = False
                p.value = fix_value
        return
    def map_freepars(self,freepars):
        i = 0
        for p in self.pars:
            if p.free:
                p.value = freepars[i]
                i +=1
        return
    def map_stderr(self,ses):
        i = 0
        for p in self.pars:
            if p.free:
                p.stderr = ses[i]
                p.tvalue = p.value / p.stderr if p.stderr != 0 else np.nan
                i +=1
        return
    def restart_from(self,scn_name='ref'):
        origin = pd.read_csv('output/params_'+scn_name+'.csv')
        print(origin)
        for p in self.pars:
            p.value = origin.loc[(origin['group']==p.group) & (origin['label']==p.label),'value'].values[0]
        return

    def parse(self):
        self.set_groups()
        n_uh = 10
        par_groups = []
        for g in self.groups:
            if g=='uh':
                Sig = np.zeros((n_uh,n_uh))
                f = np.zeros((n_uh,n_uh))
                p = np.array([p.value for p in self.pars if p.group==g])
                free = np.array([p.free for p in self.pars if p.group==g])
                k = 0
                for i in range(n_uh):
                    for j in range(i,n_uh):
                        Sig[i,j] = p[k]
                        f[i,j] = free[k]
                        if i==j:
                            if f[i,j]:
                                Sig[i,j] = np.exp(Sig[i,j])**2
                            else:
                                Sig[i,j] = 1
                        else :
                            if f[i,j]:
                                Sig[i,j] = np.tanh(Sig[i,j])
                            else :
                                Sig[i,j] = 0
                        k +=1
                # pass again for off diagonals
                for i in range(n_uh):
                    for j in range(i,n_uh):
                        if i!=j:
                            Sig[i,j] = Sig[i,j] * np.sqrt(Sig[i,i]) * np.sqrt(Sig[j,j])
                # now copy off diagonals
                for i in range(n_uh):
                    for j in range(i,n_uh):
                        if i!=j:
                            Sig[j,i] = Sig[i,j]
                # take choleski
                self.Sig = Sig
                self.L = np.linalg.cholesky(Sig)
                for i in range(n_uh):
                    for j in range(i,n_uh):
                        if i==j:
                            if f[i,j]==False:
                                self.L[i,j] = 0
                par_groups.append(self.L)
            else :
                par_groups.append(np.array([p.value for p in self.pars if p.group==g]))
        return dict(zip(self.groups,par_groups))
        
    def table(self):
        self.set_groups()
        tab = pd.DataFrame(index = np.arange(self.npars()), columns = ['group','label','value','stderr','tvalue','free'])
        for i,p in enumerate(self.pars):
            for c in tab.columns:
                tab.loc[i,c] = getattr(p,c)
        return tab

class logit:
    def __init__(self, with_uh = False):
        self.shifters = {}
        self.n_scns = 4
        self.with_uh = with_uh
        return
    def set_params(self, pars):
        self.pars = pars
        return
    def set_shifters(self, group, data):
        self.shifters[group] = data
        return
    def set_choices(self,choices):
        self.cs = choices
        self.n_obs = self.cs.shape[0]
        self.n_resp = int(self.n_obs / self.n_scns)
        return
    def set_attributes(self, attributes, attr_labels, attr_ctr):
        self.attr_labels = attr_labels
        self.attributes = attributes
        self.attr_ctr = attr_ctr
        self.n_attr = 10
        self.n_choices = 3
        return
    def set_draws(self, n_d, seed = 1234):
        if self.with_uh:
            self.n_d = n_d
            np.random.seed(seed = seed)
            self.draws = np.random.normal(size=(self.n_d, self.n_resp, self.n_attr))
            self.draws = np.where(self.draws>3,3,self.draws)
            self.draws = np.where(self.draws<-3,-3,self.draws)
        else :
            self.n_d = 1
            self.draws = None
        return
    def utility(self,alphas):
        if self.with_uh:
            us = np.zeros((self.n_obs,self.n_choices, self.n_d))
        else :
            us = np.zeros((self.n_obs,self.n_choices))
        ctr = list(self.attr_ctr.values())
        etas =  np.zeros((self.n_obs,self.n_attr))
        attributes = np.zeros((self.n_obs,self.n_choices,self.n_attr))
        for k,g in enumerate(self.attr_labels):
            etas[:,k] = np.sum(self.shifters[g] * alphas[g], axis=1)
            attributes[:,:,k] = self.attributes[g]
        if self.with_uh:
            alphas = get_alphas_uh(etas, self.draws, alphas['uh'], np.array(ctr,dtype='int64'))
            us = get_utility_uh(alphas, attributes)
        else :
            alphas = etas[:,:]
            for g in range(self.n_attr):
                if ctr[g]!=0:
                    alphas[:,g] = ctr[g]*np.exp(alphas[:,g])
            us = get_utility_no(alphas, attributes)
        return us
    def loglike(self, theta):
        self.pars.map_freepars(theta)
        alphas = self.pars.parse()
        us = self.utility(alphas)
        if self.with_uh:
            cps = cprob_uh(us,self.cs)
            ps = prob_uh(cps)
        else :
            cps = cprob_no(us,self.cs)
            ps = prob_no(cps)
        ll = np.sum(np.log(ps)) / self.n_resp
        return ll
    def loglike_i(self, theta):
        self.pars.map_freepars(theta)
        alphas = self.pars.parse()
        us = self.utility(alphas)
        if self.with_uh:
            cps = cprob_uh(us,self.cs)
            ps = prob_uh(cps)
        else :
            cps = cprob_no(us,self.cs)
            ps = prob_no(cps)
        return np.log(ps)
    def callback(self,theta):
        ll = self.loglike(theta)
        print('iter = ',self.iter, ', loglike = ',ll)
        print('current theta:')
        print(theta)
        self.iter +=1

    def doestimation(self, algorithm='BFGS'):
        init_theta = self.pars.extract_freepars()
        self.iter = 0
        negloglike = lambda theta: -self.loglike(theta)
        opt = minimize(negloglike, init_theta,method=algorithm, callback=self.callback, options={'disp': True})
        self.pars.map_freepars(opt.x)
        return
    def get_gradients(self):
        eps = 1e-6
        n_resp = self.n_resp
        nfreepars = self.pars.nfreepars()
        gradients = np.zeros((n_resp, nfreepars))
        theta = self.pars.extract_freepars()
        ll_ref = self.loglike_i(theta)
        for p in range(nfreepars):
            theta_p = np.copy(theta)
            theta_p[p] += eps
            ll_perturbed = self.loglike_i(theta_p)
            gradients[:, p] = (ll_perturbed - ll_ref) / eps
        self.pars.map_freepars(theta)
        return gradients
    def covariance(self):
        gradients = self.get_gradients()
        B = gradients.T @ gradients
        Cov = np.linalg.inv(B)
        ses = np.sqrt(np.diag(Cov))
        self.pars.map_stderr(ses)
        return
    def estimate(self, scn_name = 'ref',algorithm='BFGS'):
        self.doestimation(algorithm=algorithm)
        self.covariance()
        print(self.pars.table())
        self.pars.table().to_csv('output/params_'+scn_name+'.csv')
        return
    def get_thetas_uh(self,alphas):
        ctr = list(self.attr_ctr.values())
        etas =  np.zeros((self.n_obs,self.n_attr))
        attributes = np.zeros((self.n_obs,self.n_choices,self.n_attr))
        
        for k,g in enumerate(self.attr_labels):
            etas[:,k] = np.sum(self.shifters[g] * alphas[g], axis=1)
            attributes[:,:,k] = self.attributes[g]

        thetas = get_alphas_uh(etas, self.draws, alphas['uh'], np.array(ctr,dtype='int64'))
        return thetas
    def cond_weights(self,alphas):
        us = self.utility(alphas)
        cps = cprob_uh(us,self.cs)
        w = get_weights(cps)
        return w
    def get_cond_distribution(self):
        theta = self.pars.extract_freepars()
        self.pars.map_freepars(theta)
        alphas = self.pars.parse()
        
        thetas = self.get_thetas_uh(alphas)
        w = self.cond_weights(alphas)

        thetas_hat = get_thetas_hat(thetas,w)      
        return thetas_hat

@njit(float64[:](float64[:,:]),cache=True)
def prob_uh(cps):
    n_d = cps.shape[1]
    n_obs = cps.shape[0]
    n_scn  = 4
    n_resp = int(n_obs / n_scn)
    ps = np.zeros(n_resp,dtype='float64')
    ii = 0
    for i in range(n_resp):
        for d in range(n_d):
            ps[i] += np.prod(cps[ii:ii+n_scn,d])
        ii += n_scn
        ps[i] = max(ps[i],1e-12)/n_d
    return ps

def prob_no(cps):
    n_obs = cps.shape[0]
    n_scn  = 4
    n_resp = int(n_obs / n_scn)
    ps = np.zeros(n_resp,dtype='float64')
    ii = 0
    for i in range(n_resp):
        ps[i] = max(np.prod(cps[ii:ii+n_scn]),1e-12)
        ii += n_scn
    return ps

@njit(float64[:,:](float64[:,:,:], int64[:]),cache=True)
def cprob_uh(us,cs):
    n_d = us.shape[0]
    n_obs = us.shape[1]
    cps = np.zeros((n_obs, n_d),dtype='float64')
    for i in range(n_obs):
        for d in range(n_d):
            den = np.sum(np.exp(us[d,i,:]))
            cps[i, d] = np.exp(us[d,i,cs[i]])/max(den,1e-12)
    return cps

@njit(float64[:](float64[:,:], int64[:]),cache=True)
def cprob_no(us,cs):
    n_obs = us.shape[0]
    cps = np.zeros(n_obs,dtype='float64')
    for i in range(n_obs):
            cps[i] = np.exp(us[i,cs[i]])/np.sum(np.exp(us[i,:]))
    return cps

@njit(float64[:,:,:](float64[:,:,:], float64[:,:,:]), parallel=True,cache=True)
def get_utility_uh(alphas, attributes):
     n_d = alphas.shape[0]
     n_attr = alphas.shape[2]
     n_obs = attributes.shape[0]
     n_choices = attributes.shape[1]
     us = np.zeros((n_d,n_obs,n_choices),dtype='float64')
     for d in range(n_d):
         for i in range(n_obs):
            for c in range(n_choices):
                for g in range(n_attr):
                    us[d,i,c] += attributes[i,c,g] * alphas[d,i,g]
     return us

@njit(float64[:,:](float64[:,::], float64[:,:,:]), parallel=True,cache=True)
def get_utility_no(alphas, attributes):
     n_obs = alphas.shape[0]
     n_attr = alphas.shape[1]
     n_choices = attributes.shape[1]
     us = np.zeros((n_obs,n_choices),dtype='float64')
     for i in range(n_obs):
        for c in range(n_choices):
            for g in range(n_attr):
                us[i,c] += attributes[i,c,g] * alphas[i,g]
     return us

@njit(float64[:,:,:](float64[:,:], float64[:,:,:], float64[:,:],int64[:]), parallel=True,cache=True)
def get_alphas_uh(etas, draws, L, c):
     n_d = draws.shape[0]
     n_obs = etas.shape[0]
     n_resp = draws.shape[1]
     n_attr = draws.shape[2]
     n_scns = 4
     alphas = np.zeros((n_d,n_obs,n_attr),dtype='float64')
     for d in range(n_d):
         ii = 0
         for i in range(n_resp):
            for k in range(n_attr):
                for s in range(n_scns):
                    alphas[d,ii+s,k] = etas[ii+s,k]
                for j in range(n_attr): 
                    for s in range(n_scns):
                        alphas[d,ii+s,k] += L[k,j] * draws[d,i,j]
                if c[k]!=0:
                    for s in range(n_scns):
                        alphas[d,ii+s,k] = c[k]*np.exp(alphas[d,ii+s,k])               
            ii += n_scns
     return alphas

@njit(float64[:,:](float64[:,:]),cache=True)
def get_weights(cps):
    n_d = cps.shape[1]
    n_obs = cps.shape[0]
    n_scn  = 4
    n_resp = int(n_obs / n_scn)
    w_resp = np.zeros((n_d,n_resp),dtype='float64')
    w_obs = np.zeros((n_d,n_obs),dtype='float64')
    
    #compute weights
    ii = 0
    for i in range(n_resp):
        for d in range(n_d):
            w_resp[d,i] = np.prod(cps[ii:ii+n_scn,d])
        d_sum = np.sum(w_resp[:,i])
        for d in range(n_d):
            w_resp[d,i] = w_resp[d,i]/d_sum
        ii += n_scn

    #expand weights
    ii = 0
    for i in range(n_resp):
        for d in range(n_d):
            w_obs[d,ii:ii+n_scn] = w_resp[d,i]
        ii += n_scn
    return w_obs

@njit(float64[:,:](float64[:,:,:], float64[:,:]), parallel=True,cache=True)
def get_thetas_hat(thetas,w):
    n_d = thetas.shape[0]
    n_obs = thetas.shape[1]
    n_attr = thetas.shape[2]
    thetas_hat = np.zeros((n_obs,n_attr),dtype='float64')
    for i in range(n_obs):
        for j in range(n_attr):
            for d in range(n_d):
                thetas_hat[i,j] += w[d,i]*thetas[d,i,j]
    return thetas_hat