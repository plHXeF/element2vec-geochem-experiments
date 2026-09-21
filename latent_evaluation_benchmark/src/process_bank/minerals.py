"""Oxide masses from stoichiometric formula decomposition, not hand-set vectors."""
import re
import numpy as np
# Explicit oxide stoichiometry preserves oxidation state and bound H2O.
FORMULAS={'Quartz':{'SiO2':1},'Ab':{'Na2O':.5,'Al2O3':.5,'SiO2':3},'An':{'CaO':1,'Al2O3':1,'SiO2':2},'Or':{'K2O':.5,'Al2O3':.5,'SiO2':3},'Fo':{'MgO':2,'SiO2':1},'Fa':{'FeO':2,'SiO2':1},'En':{'MgO':1,'SiO2':1},'Fs':{'FeO':1,'SiO2':1},'Di':{'CaO':1,'MgO':1,'SiO2':2},'Hd':{'CaO':1,'FeO':1,'SiO2':2},'Magnetite':{'FeO':1,'Fe2O3':1},'Ilmenite':{'FeO':1,'TiO2':1},'Apatite':{'CaO':5,'P2O5':1.5,'H2O':.5}}
def molecular_mass(formula,weights):return sum(weights[e]*(float(n) if n else 1) for e,n in re.findall(r'([A-Z][a-z]?)([0-9.]*)',formula))
def composition(name,cfg):
 parts={o:n*molecular_mass(o,cfg['atomic_weights']) for o,n in FORMULAS[name].items()};total=sum(parts.values());return {o.lower():m/total for o,m in parts.items()}
def matrix(cfg):
 names=list(FORMULAS);a=np.array([[composition(n,cfg).get(o,0) for o in cfg['major']] for n in names]);a/=a.sum(axis=1,keepdims=True);return names,a
