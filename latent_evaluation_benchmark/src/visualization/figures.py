import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FuncFormatter
from src.common import ROOT,NAMES
COLORS=dict(zip(NAMES,['#666666','#269a83','#a18a32','#4166b0','#d9713e','#a1528c']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','axes.titleweight':'bold'})
def save(fig,path):
 Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
def draw_decodability():
 frame=pd.read_parquet(ROOT/'outputs/decodability/metrics.parquet');targets=['rock_group','rock_origin','SiO2','Mg_number','SigmaREE10'];titles=['Rock group','Rock origin','SiO2 (input removed)','Mg# (Mg/Fe inputs removed)','Sum REE: 10-field panel'];fig,axs=plt.subplots(2,3,figsize=(16,9));axs=axs.ravel()
 def panel(ax,target):
  metric='balanced_accuracy' if target.startswith('rock_') else 'R2'
  for name in NAMES:
   g=frame[(frame.target==target)&(frame.representation==name)&(frame.status=='ok')].sort_values('parameter_count');ax.plot(np.log10(g.parameter_count),g[metric],'-o',color=COLORS[name],label=name,ms=4,lw=1.6)
  ax.set_xlabel('log10(trainable probe parameters)');ax.set_ylabel('Balanced accuracy' if metric=='balanced_accuracy' else 'Test R²');ax.set_title(titles[targets.index(target)]);ax.grid(alpha=.15)
 for ax,target in zip(axs,targets):
  panel(ax,target);single,sa=plt.subplots(figsize=(7.5,5));panel(sa,target);sa.legend(ncol=3,fontsize=8);save(single,ROOT/'outputs/decodability/curves'/f'{target}.png')
  for name in NAMES:
   single,sa=plt.subplots(figsize=(6,4));g=frame[(frame.target==target)&(frame.representation==name)].sort_values('parameter_count');m='balanced_accuracy' if target.startswith('rock_') else 'R2';sa.plot(np.log10(g.parameter_count),g[m],'-o',color=COLORS[name]);sa.set(title=f'{name} · {target}',xlabel='log10(probe parameters)',ylabel=m);save(single,ROOT/'outputs/decodability/curves'/f'{target}_{name}.png')
 axs[-1].axis('off');handles,labels=axs[0].get_legend_handles_labels();axs[-1].legend(handles,labels,loc='center',frameon=False);axs[-1].text(.05,.08,'Fixed pretrained models; new probe split.\nAll samples were exposed in pretraining.\nLinear → 32 → 128/64 → 512/256/128.\nNo aggregate latent score.',transform=axs[-1].transAxes,fontsize=10)
 fig.suptitle('Decodability versus probe complexity',fontsize=18);fig.tight_layout(rect=(0,0,1,.95));p=ROOT/'outputs/figures/decodability_overview.png';save(fig,p)
 import shutil
 shutil.copy2(p,ROOT/'outputs/decodability/decodability_overview.png')
 for target in targets:
  dest=ROOT/'outputs/figures/decodability_tasks';dest.mkdir(exist_ok=True);shutil.copy2(ROOT/'outputs/decodability/curves'/f'{target}.png',dest/f'{target}.png')
def draw_sufficiency():
 r=pd.read_parquet(ROOT/'outputs/sufficiency/masked_recovery.parquet');n=pd.read_parquet(ROOT/'outputs/sufficiency/nuisance.parquet');d=pd.read_parquet(ROOT/'outputs/decodability/metrics.parquet');fig,ax=plt.subplots(1,3,figsize=(17,5))
 groups=['major','trace','REE','isotope','age/derived']
 for name in NAMES:
  g=r[(r.representation==name)&(r.status=='ok')].groupby('group').R2.median().reindex(groups);ax[0].plot(range(len(groups)),g,'-o',label=name,color=COLORS[name]);wanted=d[(d.representation==name)&d.target.eq('rock_group')&d.probe.eq('Linear')].balanced_accuracy.iloc[0];leak=n[n.representation.eq(name)&n.target.eq('data_source')].balanced_accuracy.iloc[0];ax[1].scatter(leak,wanted,color=COLORS[name],s=65);ax[1].annotate(name,(leak,wanted),xytext={'V2':(8,-19),'RoPEz':(-38,13),'RoPEgeo':(-14,30)}.get(name,(5,3)),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':COLORS[name],'lw':.6} if name in NAMES[3:] else None);ng=n[n.representation.eq(name)&n.target.eq('same_source_knn_enrichment')];ax[2].bar(name,ng.enrichment.iloc[0],color=COLORS[name])
 ax[0].set_xticks(range(len(groups)),['major','trace','REE','isotope*','age/derived*'],rotation=20);ax[0].set_yscale('symlog',linthresh=.5);ax[0].axhline(0,color='grey',lw=.7);ax[0].set_title('Linear recovery / external prediction*');ax[0].set_ylabel('Median field R² (symlog; linear near zero)');ax[1].margins(y=.2);ax[0].legend(fontsize=8,ncol=2);ax[1].set(xlabel='Source leakage: linear balanced accuracy',ylabel='Wanted: rock-group linear balanced accuracy',title='Wanted information versus nuisance');ax[2].set(title='Same-source Euclidean kNN enrichment',ylabel='Observed / random frequency baseline');ax[2].tick_params(axis='x',rotation=35);fig.tight_layout();save(fig,ROOT/'outputs/figures/sufficiency_vs_leakage.png')
def draw_semantic():
 data=pd.read_parquet(ROOT/'outputs/semantic_geometry/metrics.parquet');tasks=list(data.target.unique());fig,ax=plt.subplots(1,2,figsize=(17,6))
 for axis,metric in zip(ax,['triplet_accuracy','smoothness']):
  for name in NAMES:
   g=data[data.representation.eq(name)].set_index('target').reindex(tasks);axis.plot(range(len(tasks)),g[metric],'-o',label=name,color=COLORS[name],ms=4)
  axis.set_xticks(range(len(tasks)),tasks,rotation=40,ha='right');axis.set_ylabel(metric);axis.grid(alpha=.2);axis.axhline(.5 if metric=='triplet_accuracy' else 0,color='grey',ls=':',lw=1)
 ax[0].set_title('Matched continuous and hierarchy triplets');ax[1].set_title('kNN smoothness relative to 100 permutations');ax[1].legend(ncol=3,fontsize=8);fig.suptitle('Native representation Euclidean geometry',fontsize=17);fig.tight_layout();save(fig,ROOT/'outputs/figures/semantic_geometry.png')
def draw_process():
 pair=pd.read_parquet(ROOT/'outputs/tables/paired_statistics.parquet');processes=['Mixing','Fractionation','PartialMelting','Alteration','Composite','heterogeneous','same-process'];metrics=['tangent_coherence','simplicity','local_rank1','no_fold','normalized_next_state_error','boundary_f1','composition_error'];titles=['Tangent coherence','Simplicity','Local rank-1','No-fold','Operator error ↓','Boundary F1','Composition error ↓'];fig,axes=plt.subplots(7,7,figsize=(23,19),sharex=False)
 for i,process in enumerate(processes):
  for j,metric in enumerate(metrics):
   ax=axes[i,j];subset=pair[(pair.process==process)&(pair.metric==metric)&pair.baseline.isin(['GeoNorm','ZVector'])]
   if subset.empty:ax.text(.5,.5,'Not applicable',ha='center',va='center',transform=ax.transAxes,color='#888888');ax.set_xticks([]);ax.set_yticks([])
   else:
    for k,name in enumerate(NAMES[3:]):
     for b,base in enumerate(['GeoNorm','ZVector']):
      for s,source in enumerate(['synthetic','semi-synthetic']):
       g=subset[(subset.representation==name)&(subset.baseline==base)&(subset.source_type==source)]
       if g.empty:continue
       r=g.iloc[0];y=k+(b-.5)*.23+(s-.5)*.09;ax.errorbar(r.median_delta,y,xerr=[[max(0,r.median_delta-r.ci_low)],[max(0,r.ci_high-r.median_delta)]],fmt='o' if base=='GeoNorm' else 's',ms=4,mfc=COLORS[name] if source=='synthetic' else 'white',mec=COLORS[name],ecolor=COLORS[name],lw=1)
    if subset[['ci_low','ci_high']].abs().to_numpy().max()>10:ax.set_xscale('symlog',linthresh=.02)
    ax.axvline(0,color='grey',lw=.8);ax.set_yticks(range(3),NAMES[3:] if j==0 else []);ax.grid(axis='x',alpha=.15);ax.xaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.3g}'))
   if i==0:ax.set_title(titles[j],fontsize=11)
   if j==0:ax.set_ylabel(process,fontsize=11)
 fig.suptitle('Paired model − baseline effects (positive = improvement)\nCircles: GeoNorm · Squares: ZVector · Filled: synthetic · Open: semi-synthetic · 95% cluster bootstrap CI',fontsize=16);fig.tight_layout(rect=(0,0,1,.94));save(fig,ROOT/'outputs/figures/process_benchmark.png')
 op=pd.read_parquet(ROOT/'outputs/process/operator_metrics.parquet').query("split == 'test'");matrix=op.pivot_table(index='representation',columns='process',values='normalized_next_state_error',aggfunc='median').reindex(NAMES);fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(matrix.to_numpy(),cmap='magma_r',aspect='auto');ax.set_xticks(range(len(matrix.columns)),matrix.columns);ax.set_yticks(range(len(matrix)),matrix.index)
 for i in range(len(matrix)):
  for j in range(len(matrix.columns)):ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',color='white' if matrix.iloc[i,j]>matrix.max().max()/2 else 'black')
 ax.set_title('Held-out-episode next-state error / train latent RMS radius');fig.colorbar(im,ax=ax);save(fig,ROOT/'outputs/figures/process_operator_error.png')
 hist=pd.read_parquet(ROOT/'outputs/tables/convergence.parquet');fig,ax=plt.subplots(1,2,figsize=(12,5))
 for name in NAMES[3:]:
  g=hist[hist.representation==name].groupby('bank_total_episodes').ci_half_width;ax[0].plot(g.max().index,g.max(),'-o',label=name,color=COLORS[name]);ax[1].plot(g.median().index,g.median(),'-o',color=COLORS[name])
 for a,title in zip(ax,['Maximum CI half-width','Median CI half-width']):a.axhline(.02,color='black',ls='--');a.set(xlabel='Total saved episodes (adaptive families only)',ylabel='Paired effect 95% CI half-width',title=title);a.grid(alpha=.2)
 ax[0].set_yscale('symlog',linthresh=.02);ax[0].legend();fig.tight_layout();save(fig,ROOT/'outputs/figures/monte_carlo_convergence.png')
def all_figures():
 (ROOT/'outputs/figures').mkdir(exist_ok=True);draw_decodability();draw_sufficiency();draw_semantic();draw_process()
