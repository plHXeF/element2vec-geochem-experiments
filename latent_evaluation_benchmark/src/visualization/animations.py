"""One PCA3 per representation, fitted once on a frozen shared-bank reference."""
import json,pickle,shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter,PillowWriter
from src.common import ROOT,NAMES,save_json,digest
from src.visualization.figures import COLORS

def render_animations(cfg):
 bank=ROOT/'cache/process_bank';index=json.loads((bank/'index.json').read_text());lookup={r['episode_id']:r for r in index};fs=json.loads((bank/'schema.json').read_text())['fields'];out=ROOT/'outputs/animations';out.mkdir(exist_ok=True);metrics=pd.read_parquet(ROOT/'outputs/process/trajectory_metrics.parquet');pcas={};reference_ids=sorted(lookup,key=lambda s:digest([cfg['seed'],s]))[:2000]
 for name in NAMES:
  path=ROOT/'cache'/f'visualization_pca_{name}.pkl'
  if path.exists():pcas[name]=pickle.loads(path.read_bytes());continue
  arrays=[]
  for eid in reference_ids:arrays.append(np.load(ROOT/'cache/process_latents'/name/(eid+'.npz'))['z'][::4])
  x=np.vstack(arrays);pca=PCA(n_components=3,svd_solver='full').fit(x);path.write_bytes(pickle.dumps(pca));pcas[name]=pca
 save_json(out/'pca_reference.json',{'reference_episode_ids':reference_ids,'reference_id_hash':digest(reference_ids),'state_stride':4,'basis_scope':'fixed across all animations, one basis per representation','metrics_use_this_basis':False})
 selected=[];rng=np.random.default_rng(cfg['seed'])
 for kind in ['Mixing','Fractionation','Alteration','Composite']:
  pool=metrics[(metrics.kind==kind)&metrics.representation.eq('GeoNorm')].sort_values(['simplicity','episode_id']);median=pool.iloc[len(pool)//2].episode_id
  v=metrics[(metrics.kind==kind)&metrics.representation.eq('V2')][['episode_id','simplicity']].merge(pool[['episode_id','simplicity']],on='episode_id',suffixes=('_model','_baseline'));v['difference']=abs(v.simplicity_model-v.simplicity_baseline);disagree=v.sort_values(['difference','episode_id']).iloc[-1].episode_id;random=str(rng.choice(sorted(pool.episode_id)))
  for rule,eid in [('median',median),('disagreement',disagree),('random',random)]:selected.append({'kind':kind,'rule':rule,'episode_id':eid})
 save_json(out/'selection.json',{'rules':{'median':'middle GeoNorm simplicity, tie break episode ID','disagreement':'max absolute V2 vs GeoNorm simplicity difference','random':'master-seed uniform draw'},'episodes':selected})
 for choice in selected:
  eid=choice['episode_id'];filename=f"{choice['kind']}_{choice['rule']}";video=out/(filename+'.mp4' if shutil.which('ffmpeg') else filename+'.gif')
  if video.exists():continue
  row=lookup[eid];raw=np.load(bank/row['state_file']);states=raw['states'];latent={n:pcas[n].transform(np.load(ROOT/'cache/process_latents'/n/(eid+'.npz'))['z']) for n in NAMES};fig=plt.figure(figsize=(15,11));major=fig.add_subplot(331);trace=fig.add_subplot(332);info=fig.add_subplot(333);info.axis('off');axes={n:fig.add_subplot(3,3,j+4,projection='3d') for j,n in enumerate(NAMES)}
  major_fields=['sio2','al2o3','mgo','cao'];trace_fields=['la_ppm','sr_ppm','ni_ppm'];mlines={};tlines={};trails={};dots={}
  for f,color in zip(major_fields,['#4166b0','#d9713e','#269a83','#a1528c']):
   j=fs.index(f);major.plot(states[:,j],alpha=.15,color=color);mlines[f],=major.plot([],[],label=f,color=color)
  for f,color in zip(trace_fields,['#4166b0','#d9713e','#269a83']):
   j=fs.index(f);trace.plot(np.maximum(states[:,j],1e-9),alpha=.15,color=color);tlines[f],=trace.plot([],[],label=f,color=color)
  major.set(title='Raw major chemistry',xlabel='trajectory step',ylabel='wt%');trace.set(title='Raw trace chemistry',xlabel='trajectory step',ylabel='ppm',yscale='log');major.legend(fontsize=8);trace.legend(fontsize=8)
  for boundary in row['stage_boundaries']:major.axvline(boundary,color='grey',ls=':');trace.axvline(boundary,color='grey',ls=':')
  for n,ax in axes.items():
   z=latent[n];ax.plot(*z.T,color=COLORS[n],alpha=.15);trails[n],=ax.plot([],[],[],color=COLORS[n],lw=2);dots[n],=ax.plot([],[],[],'o',color=COLORS[n],ms=6)
   for b in row['stage_boundaries']:ax.scatter(*z[b],marker='x',color='black',s=40)
   for j,lim in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):
    low,high=np.min(z[:,j]),np.max(z[:,j]);pad=max((high-low)*.15,.01);lim(low-pad,high+pad)
   ax.set(title=n,xlabel='PC1',ylabel='PC2',zlabel='PC3');ax.tick_params(labelsize=7)
  text=info.text(0,1,'',va='top',fontsize=10,family='monospace');arrows=[]
  def update(k):
   nonlocal arrows
   for art in arrows:art.remove()
   arrows=[]
   for f,line in mlines.items():line.set_data(np.arange(k+1),states[:k+1,fs.index(f)])
   for f,line in tlines.items():line.set_data(np.arange(k+1),np.maximum(states[:k+1,fs.index(f)],1e-9))
   stage=min(int(np.searchsorted(row['stage_boundaries'],k,side='right')),len(row['process_sequence'])-1);begin=0 if stage==0 else row['stage_boundaries'][stage-1];g=row['stage_grids'][stage];local=min(k-begin,len(g)-1);parameter=g[local];name=row['process_sequence'][stage];chem='\n'.join(f'{f:>8}: {states[k,fs.index(f)]:.3g}' for f in major_fields+trace_fields)
   text.set_text(f"{choice['kind']} / {choice['rule']}\n{row['source_type']}\nStage {stage+1}: {name}\n{'F' if name in ['Fractionation','PartialMelting'] else 't'} = {parameter:.4f}\nStep {k}/{len(states)-1}\n\n{chem}\n\nPCA for display only.")
   for n,ax in axes.items():
    z=latent[n];trails[n].set_data_3d(*z[:k+1].T);dots[n].set_data_3d(*z[k:k+1].T)
    if k<len(z)-1:
     direction=z[k+1]-z[k];arrows.append(ax.quiver(*z[k],*direction,color=COLORS[n],length=2,normalize=False))
   return []
  fig.suptitle(f"Shared physical episode: {eid}",fontsize=15);fig.tight_layout(rect=(0,0,1,.96))
  animation=FuncAnimation(fig,update,frames=len(states),interval=100,blit=False);animation.save(video,writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-pix_fmt','yuv420p']) if shutil.which('ffmpeg') else PillowWriter(fps=10),dpi=100)
  for k in sorted(set([0,len(states)//2,len(states)-1]+row['stage_boundaries'])):update(k);fig.savefig(out/f'{filename}_frame{k:03d}.png',dpi=110)
  plt.close(fig);print('Animation',video.name,flush=True)
