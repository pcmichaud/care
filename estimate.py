from model import param, params, logit, survey
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)

# load data
sago = survey()
sago.load()
attributes = sago.extract_attributes()
cs = sago.extract_choices()

print(len(sago.data))

# initialize parameters
init_pars = params()
#shifter_labels = ['constant','needs_level','prob_help_adl']
#shifter_labels = ['constant','needs_level','prob_help_adl','help_adl_miss']
shifter_labels = ['constant']
for s in sago.attr_labels:
     for h in shifter_labels:
          if s in ['svc_inf','svc_avq','svc_avd','cons','wait'] and h=='constant':
               p = param(s,h,value = -2)
          else :
               p = param(s,h,value = 0)
          init_pars.add_param(p) 
shifters = sago.extract_shifters(shifter_labels)     
n_attr = len(sago.attr_labels) 

model = logit(with_uh = False)
model.set_params(init_pars)
for s in sago.attr_labels:  
    model.set_shifters(s,shifters)
model.set_choices(cs)

ctr = [0,0,0,0,0,1,1,1,1,-1]
ctrs = dict(zip(sago.attr_labels,ctr))
model.set_attributes(attributes, sago.attr_labels, ctrs)

model.estimate()

init_pars.restart_from('ref')
for d in range(n_attr):
     for u in range(d,n_attr):
         if d==u:
               if d not in [3,4,8]:
                    p = param('uh','ln_sig_'+str(d)+str(u), value = np.log(0.5),free=True)
               else :
                    p = param('uh','ln_sig_'+str(d)+str(u), value = -10,free=False)
         else:
               #if (d not in [0,1,2,3,4,6,7,8,9]) and (u not in [0,1,2,3,4,6,7,8,9]):
                    p = param('uh','rho_'+str(d)+str(u),value = 0,free=False)
               #else :
               #     p = param('uh','rho_'+str(d)+str(u),value = 0,free=True)
         init_pars.add_param(p)

print(init_pars.table())

model = logit(with_uh = True)
model.set_params(init_pars)
for s in sago.attr_labels:  
    model.set_shifters(s,shifters)
model.set_choices(cs)

ctr = [0,0,0,0,0,1,1,1,1,-1]
ctrs = dict(zip(sago.attr_labels,ctr))
model.set_attributes(attributes, sago.attr_labels, ctrs)
model.set_draws(n_d = 25)

model.estimate()




