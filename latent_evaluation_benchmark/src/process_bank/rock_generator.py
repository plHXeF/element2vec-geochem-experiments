import numpy as np
from src.process_bank.minerals import matrix
class RockGenerator:
 def __init__(self,cfg,prior,fields):self.cfg=cfg;self.prior=prior;self.fields=fields;self.names,self.m=matrix(cfg)
 def sample(self,rng):
  groups=self.cfg['process']['modal_centers'];kind=rng.choice(list(groups));center=np.array(groups[kind]);modal=np.zeros(6);on=center>0;modal[on]=rng.dirichlet(self.cfg['process']['dirichlet_concentration']*center[on]);xan,xfo=rng.uniform(0,1,2);px=rng.dirichlet(np.ones(4));accessory=rng.dirichlet(np.ones(3));w=np.zeros(len(self.names))
  def add(name,v):w[self.names.index(name)]+=v
  add('Quartz',modal[0]);add('Ab',modal[1]*(1-xan));add('An',modal[1]*xan);add('Or',modal[2]);
  for n,v in zip(['En','Fs','Di','Hd'],px):add(n,modal[3]*v)
  add('Fo',modal[4]*xfo);add('Fa',modal[4]*(1-xfo))
  for n,v in zip(['Magnetite','Ilmenite','Apatite'],accessory):add(n,modal[5]*v)
  d=dict(zip(self.cfg['major'],100*w@self.m));d.update({'loi':0.,'co2':0.});d.update(self.prior.sample(rng));d.update(self.prior.ree(rng));return np.array([d.get(f,np.nan) for f in self.fields]),{'rock_family':kind,'modal_mass_fractions':modal.tolist(),'XAn':xan,'XFo':xfo,'pyroxene':px.tolist(),'accessory':accessory.tolist()}
