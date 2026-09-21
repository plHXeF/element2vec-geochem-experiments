from src.common import SOURCE_ROOT
import json,platform,sys,importlib.metadata,pickle,html
from pathlib import Path
import pandas as pd
import numpy as np
from src.common import ROOT,NAMES,save_json,sha,digest

def markdown(frame):
 if frame.empty:return 'No evaluable records.\n'
 def cell(v):
  if isinstance(v,float):return f'{v:.4f}' if np.isfinite(v) else 'NA'
  return str(v)
 return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def manifest(cfg):
 data=json.loads((ROOT/'cache/data_manifest.json').read_text());bank=json.loads((ROOT/'cache/process_bank/schema.json').read_text());audit=json.loads((ROOT/'outputs/audit/compatibility.json').read_text());files={}
 for d in ['src','configs','tests']:
  for p in (SOURCE_ROOT/d).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json']:files[str(p.relative_to(SOURCE_ROOT))]=sha(p)
 packages={}
 for name in ['numpy','pandas','scipy','scikit-learn','torch','matplotlib','pyarrow']:
  try:packages[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:packages[name]='local vendor'
 result={'benchmark_version':'0.1','config':cfg,'config_sha256':sha(ROOT/'configs/benchmark.json'),'master_seed':cfg['seed'],'data':data,'feature_schema_hash':digest(cfg['features']),'recovery_mask_sha256':sha(ROOT/'cache/recovery_mask.npy'),'semantic_triplets_sha256':sha(ROOT/'cache/semantic_triplets.npz'),'bank':bank,'checkpoints':[{'name':a['name'],'path':a['checkpoint'],'sha256':a['sha256']} for a in audit['experiments']],'baseline_statistics':json.loads((ROOT/'configs/fitted_baseline_statistics.json').read_text()),'source_hashes':files,'python':sys.version,'platform':platform.platform(),'packages':packages,'metric_space':'native representation; Euclidean, one scalar RMS radius only, no post-hoc metric PCA','PCA_Geo':'PCA is the definition of this requested baseline; it is not a projection applied to model metric spaces','probe_config':cfg['probe'],'evaluation_protocol':json.loads((SOURCE_ROOT/'configs/evaluation_protocol.json').read_text()),'outputs_regeneration':'python run.py render reads cached latents, tables and persisted PCA; original models are not invoked','pretraining_exposure':'all models trained on full data; new splits do not establish external generalization','operator_fit':'fixed first-stage EpisodeBank train initial IDs; stride 4; later Monte Carlo additions do not retrain operators','precision_status':json.loads((ROOT/'outputs/tables/precision_status.json').read_text())}
 result['visualization_PCA']={}
 for name in NAMES:
  path=ROOT/'cache'/f'visualization_pca_{name}.pkl';pca=pickle.loads(path.read_bytes())
  result['visualization_PCA'][name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'n_components':pca.n_components_,'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'fit_reference':json.loads((ROOT/'outputs/animations/pca_reference.json').read_text())['reference_id_hash']}
 save_json(ROOT/'outputs/run_manifest.json',result);return result

def write_report(cfg,records):
 dec=pd.read_parquet(ROOT/'outputs/decodability/metrics.parquet');rec=pd.read_parquet(ROOT/'outputs/sufficiency/masked_recovery.parquet');nuis=pd.read_parquet(ROOT/'outputs/sufficiency/nuisance.parquet');sem=pd.read_parquet(ROOT/'outputs/semantic_geometry/metrics.parquet');traj=pd.read_parquet(ROOT/'outputs/process/trajectory_metrics.parquet');op=pd.read_parquet(ROOT/'outputs/process/operator_metrics.parquet');comp=pd.read_parquet(ROOT/'outputs/composite/metrics.parquet');order=pd.read_parquet(ROOT/'outputs/composite/noncommutativity.parquet');pair=pd.read_parquet(ROOT/'outputs/tables/paired_statistics.parquet');run=manifest(cfg)
 lines=['# Latent Evaluation Benchmark v0.1','', '本报告不计算人为加权总分。所有结果比较固定模型如何重组化学信息，以及这种重组的代价。','', '## Scope and provenance', '',f"真实记录 {len(records):,}；split 为 sample-ID hash 70/15/15。过程库每种单过程 {run['bank']['per_process']:,} 条（synthetic / semi-synthetic 各半），另有 {run['bank']['composites']:,} 条 composite 和 4 条矿物固溶体校准。原三个模型没有重训、没有改写。",'', '三个模型都在全量样品及岩石标签上预训练。所有真实数据得分是 fixed-representation decodability / geometry，不能当作未见文献或地区泛化。样品随机划分保留来源重叠，便于测量 nuisance；完整文献分组外推不是本轮结论。', '', '## 1. Decodability', '', '所有 label token 始终移除。SiO2 在任何模型/基线预处理前删除 sio2；Mg# 删除 mgo/feo/fe2o3；ΣREE 删除全部 REE 输入。目标遮盖后才计算闭合、CLR 和 RoPE_geo 坐标。Mg# 定义为 100 mol(MgO)/(mol(MgO)+mol(FeO))，不假设 ferric iron 全部为 Fe2+。ΣREE10 是数据库中 10 个 REE 的完整面板之和，不冒充全 14 元素总量。','', '四档 probe 使用 GELU（隐层）、AdamW、相同两个学习率预算和 early stopping。参数量含偏置，曲线横轴是真实参数量。AUC 的参数区间因输入维度不同，不能替代原始曲线。','', '![Probe complexity](outputs/figures/decodability_overview.png)','']
 for target in dec.target.unique():
  g=dec[dec.target.eq(target)];metric='balanced_accuracy' if target.startswith('rock_') else 'R2';pivot=g.pivot(index='representation',columns='probe',values=metric).reindex(NAMES).reset_index();lines+=[f'### {target}: {metric}', '',markdown(pivot),'']
 lines+=['## 2. Information Sufficiency','', '真实 test 中共享同一随机 20% 数值字段遮盖；linear Ridge 仅从 masked-record 表示恢复原始值，超参数只看 val。isotope/age/derived 从未进入任何 encoder，因此这些组属于外部目标预测，不是对曾可见输入的字面重建。单字段支持数量与不可评估项均保存在表中；没有补造标签。','',markdown(rec[rec.status.eq('ok')].pivot_table(index='representation',columns='group',values='R2',aggfunc='median').reindex(NAMES).reset_index()),'', '来源/reference/missingness 用 linear AdamW probe。Reference 仅取 train 中至少 10 条的前 64 个文献；覆盖比例单列，不混同全词表分类。12 个 missingness cluster 仅在 train availability 上拟合。','',markdown(nuis[nuis.target.isin(['data_source','canonical_ref_id','missingness_cluster'])][['representation','target','balanced_accuracy','macro_F1','coverage']]),'', '![Sufficiency and nuisance](outputs/figures/sufficiency_vs_leakage.png)','', '## 3. Semantic Geometry','', '所有 triplet 和 kNN 距离在原生空间用 Euclidean 计算，仅允许由真实 train 拟合的单个 global RMS scale。PCA-Geo 是预先定义的基线表示；没有对模型 latent 再做 PCA 后打分。连续 triplet 尽量匹配 rock_group 和可用测项数；fallback 数量已保存。Eu 异常由 Sm/Eu/Gd 外部字段定义，三个模型都没直接读过这些字段。','',markdown(sem.pivot(index='representation',columns='target',values='triplet_accuracy').reindex(NAMES).reset_index()),'', '![Semantic geometry](outputs/figures/semantic_geometry.png)','', '## 4. Process / Law','', '每条 episode 的物理状态、参数、seed、initial-state ID、stage boundaries 和完整轨迹保存于共享 EpisodeBank。六种表示消费相同状态，模型按原预处理决定 token 是否有效。Synthetic 和 semi-synthetic 分开统计。所有合成范围均相对 train 的 p1/p99 记录 OOD，未按模型单独裁剪。','', '矿物由化学计量计算氧化物质量分数；modal 在此按质量分数解释。含 OH 的 apatite 先包含 H2O，再报告无水 oxide panel。结晶分离是受氧化物预算约束的随机矿物移除，trace 同时服从独立的 synthetic Rayleigh D；它不是经过相平衡校准的岩浆模型。部分熔融 law 指标侧重 trace。Alteration 是 sparse log-concentration law，不命名为特定自然蚀变。','', '复合路径只使用 Mixing/Fractionation/Alteration，严格把前段最终状态交给后段。Batch melt 的 F=.01 首个产物不等于源岩，因而没有强行拼成连续复合边界。','', '### Trajectory geometry', '', 'Tangent、simplicity、7-state local rank-1、no-fold 均逐 episode 保存。零位移记不可定义，不奖励为完美平滑。','', '![Process paired effects](outputs/figures/process_benchmark.png)','', '### Linear operators and composition', '', 'Ridge vector field 使用全维 latent 和明确物理控制 theta。Mixing 的 theta 额外包含该段起点 latent，并清零原始起点化学量通道；复合 rollout 中这一起点严格来自前段预测，避免真实中间状态作为条件泄漏。训练按 initial state 隔离 70/15/15，同一轨迹的 states 不跨 split。初始库的训练 episode 每 4 步取一个拟合点，test 每一步均评价。后续 Monte Carlo 增量不重新拟合算子，避免精度曲线同时改变估计对象。误差除以真实 train 的 latent RMS radius；另存相对步长误差和 R²。','',markdown(op[op.split.eq('test')].pivot_table(index='representation',columns='process',values='normalized_next_state_error',aggfunc='median').reindex(NAMES).reset_index()),'', '![Operator error](outputs/figures/process_operator_error.png)', '', 'Operator coefficient counts (including intercept; not a weighted quality score):', '', markdown(op.groupby(['representation','process']).parameter_count.first().reset_index()),'', 'Composite composition 发散时记为 1e15 train-radius 单位的失败惩罚并另存 divergence flag，不在配对比较中悄悄剔除；它从预测的中间状态继续迭代，不在 stage boundary 重置到真实 latent。Boundary 阈值只在 val composites 选取，test 容差 ±2 步；原生空间 change score 为相邻位移的 1−cos。物理 trace semigroup 校验与 learned rollout error 分开，前者成立不表示 latent 已学会该定律。','',markdown(comp[comp.split.eq('test')].groupby(['representation','composite_type'])[['boundary_f1','composition_error','rollout_diverged']].median().reset_index()),'', 'Non-commutativity 使用相同初始状态和参数进行 AB/BA，非法反序结果显式排除；原始物理浓度的 Euclidean order effect 与 latent order effect 的 Spearman 如下。混合单位的 raw distance 仅作此诊断，不作为地质距离定义。','',markdown(order),'', '### Monte Carlo and uncertainty','',f"每个 model 相对 ZVector/GeoNorm/PCA-Geo 按 episode ID 严格配对，bootstrap 2,000 次，并按重复的自然 initial-state ID 聚类。正 delta 一律代表 model 改善。精度判据为 95% CI 半宽 ≤0.02；当前超过阈值的比较 {run['precision_status']['failed_comparisons']} 项，最大半宽 {run['precision_status']['max_ci_half_width']:.4g}。达到 8,000 上限仍不满足的指标保留不确定性，不声称已收敛。",'', '![Monte Carlo](outputs/figures/monte_carlo_convergence.png)','', '### Animations','', '12 个同步动画覆盖 Mixing、Fractionation、Alteration、Composite，每类按预定义 median / max V2–GeoNorm disagreement / fixed-seed random 规则选取。每种表示使用一个固定 PCA3，所有 episode 共用该基；同时展示原始 chemistry、当前参数、过程方向和 boundary。选择清单与 basis 位于 cache 和 outputs/animations。','', '## Evidence-driven comparison','']
 # Concrete, automated prose derived from final tables, not hard-coded expectations.
 for name in NAMES[3:]:
  lines += [f'### {name}','']
  lines.append('固定 linear probe 的逐目标增量如下（分类用 balanced accuracy，回归用 R²；不是综合分数）：')
  details=[]
  for target in ['rock_group','rock_origin','SiO2','Mg_number','SigmaREE10']:
   metric='balanced_accuracy' if target.startswith('rock_') else 'R2';g=dec[dec.target.eq(target)&dec.probe.eq('Linear')].set_index('representation');value=float(g.loc[name,metric]);details.append({'target':target,'model':value,**{f'delta_vs_{b}':value-float(g.loc[b,metric]) for b in NAMES[:3]}})
  lines += ['',markdown(pd.DataFrame(details)),'']
  mix=pair[pair.representation.eq(name)&pair.baseline.eq('ZVector')&pair.process.eq('Mixing')&pair.metric.eq('simplicity')]
  for _,r in mix.iterrows():lines.append(f"- 原始 ZVector 二元混合为直线；{r.source_type} 的 simplicity 差为 {r.median_delta:.4g}，CI [{r.ci_low:.4g}, {r.ci_high:.4g}]。负差表示 encoder 将原本简单的混合关系弯曲。")
  lines.append('- 算子同属线性 Ridge 家族；768D latent 的系数数量通常明显多于传统基线。必须同时看误差和上表的参数量，不能仅凭较低误差声称 operator complexity 下降。')
  for base in ['ZVector','GeoNorm','PCAGeo']:
   comparisons=pair[pair.representation.eq(name)&pair.baseline.eq(base)];pos=comparisons[comparisons.ci_low>0];neg=comparisons[comparisons.ci_high<0]
   lines.append(f'- 相对 {base}：{len(pos)} 个分层 process/metric 的 paired CI 全在零上，{len(neg)} 个全在零下；其余不能据此判为稳定差异。这是多项探索性比较计数，不是总分或多重检验后的发现数。')
   if len(pos):r=pos.sort_values('median_delta',ascending=False).iloc[0];lines.append(f"  - 改善例：{r.process}/{r.source_type}/{r.metric}，Δ={r.median_delta:.4g}，95% CI [{r.ci_low:.4g}, {r.ci_high:.4g}]。")
   if len(neg):r=neg.sort_values('median_delta').iloc[0];lines.append(f"  - 损失例：{r.process}/{r.source_type}/{r.metric}，Δ={r.median_delta:.4g}，95% CI [{r.ci_low:.4g}, {r.ci_high:.4g}]。")
  lines.append('')
 lines+=['## Limitations and acceptance','', '固定训练 seed，没有外部重训重复；探针、合成 prior 和过程几何不等同自然过程验证。单一低误差/高平滑性不构成优胜模型结论。半合成初值需完整合法的主量面板，覆盖的是数据子集。负 trace 浓度（LOI 允许有符号）及任何氧化物超过 100 wt% 的 episode 均按原 seed 拒绝重采样；输入内容签名防止重采样后误用旧 latent。Alteration 为未闭合、固定参考质量下的浓度定律，不能把它解释成已经归一化的自然全岩产物。模型共有 34 字段，外部 REE/isotope/age 仅作 target 或 ProcessBank 扩展状态。所有 checkpoint SHA-256 在本轮前后校验，未写回原目录。','', 'Machine-readable results: CSV + Parquet + JSON in `outputs/`. Reproducibility: `outputs/run_manifest.json`. Sanity and artifact verification: `outputs/verification.json`. One command and cache-only rendering are documented in README.','']
 lines+=['## Interpretation of this run','',
 '三个 learned latent 的收益首先表现为更容易读出岩石类别，而不是普遍更忠实的化学距离。Linear rock-group balanced accuracy 约 0.662–0.665，高于 ZVector 的 0.544 和 GeoNorm 的 0.451；但三个模型的 source prediction 也更强，不能把全部提升解释为纯地质信息。', '',
 'V2 保留较强的化学恢复能力：遮盖 SiO2 后 linear R² 为 0.951，主量/微量 masked recovery 的字段中位 R² 分别为 0.646/0.354。RoPE-z 的 rock-origin linear balanced accuracy 为 0.494，高于 V2 的 0.446，但 SiO2 恢复降到 0.884；RoPE_geo+15%mask 为 0.849。三者遮盖 REE 输入后的 ΣREE10 linear R² 都接近 0，不能据此声称能可靠推断总稀土丰度。', '',
 'RoPE_geo+15%mask 更适合本轮部分合成动力学：结晶分离的归一化一步误差中位数为 0.074，低于 V2 的 0.172、RoPE-z 的 0.158、GeoNorm 的 0.262；但 ZVector 已达 0.071。GeoNorm 在 sparse log alteration 和 batch partial melting 上的一步误差仍低于三个 learned latent。这说明过程收益依赖具体坐标和定律。', '',
 '原始线性混合在 ZVector 中本来就是直线。三个 encoder 均使它弯曲：相对 ZVector，synthetic simplicity 的配对中位差约为 V2 −0.499、RoPE-z −0.532、RoPE_geo −0.298，置信区间均低于零。RoPE_geo 的扭曲较小，但没有恢复原始混合关系的线性。', '',
 '复合过程没有统一优势。异类 composite 的终点误差中位数约为 ZVector 0.224、RoPE_geo 0.413、RoPE-z 0.593、V2 0.666、PCA-Geo 0.924、GeoNorm 1.128；同类 composite 的相对次序不同。35 个 composition-error 配对区间在 8,000 条上限仍未满足目标精度，细微差异应保留判断。算子参数量在 768D latent 中大幅增加，因此不能把较低误差直接称为更低算子复杂度。', '',
 '顺序效应的幅度排序在 learned latent 中有中等保留：Spearman 约 0.43–0.55，高于本轮 GeoNorm/PCA-Geo，但低于 ZVector 的 0.60–0.68。这里的 raw order effect 是混合单位浓度的 Euclidean 距离，仅为预定义诊断。反序合法性覆盖如下；这些结论只适用于有效子集。', '',
 markdown(pd.read_parquet(ROOT/'outputs/composite/order_coverage.parquet')), '',
 '以上是本轮已保存结果的解释；它们不证明 RoPE 的单独因果效应，因为 RoPE_geo 分支同时改变数值坐标和 masking 设置，也不证明跨来源泛化。', '',
 '## Artifact links','', '[Animation gallery](outputs/animations/index.html) · [Reproduction instructions](README.md) · [Run manifest](outputs/run_manifest.json) · [Verification](outputs/verification.json)','']
 for choice in json.loads((ROOT/'outputs/animations/selection.json').read_text())['episodes']:
  name=f"{choice['kind']}_{choice['rule']}";ext='mp4' if (ROOT/'outputs/animations'/(name+'.mp4')).exists() else 'gif'
  lines.append(f"- [{name}](outputs/animations/{name}.{ext}) — `{choice['episode_id']}`")
 (ROOT/'REPORT.md').write_text('\n'.join(lines))

def animation_gallery():
 out=ROOT/'outputs/animations';choices=json.loads((out/'selection.json').read_text())['episodes'];cards=[]
 for c in choices:
  name=f"{c['kind']}_{c['rule']}";path=out/(name+'.mp4');media=f'<video controls preload="metadata" src="{name}.mp4"></video>' if path.exists() else f'<img src="{name}.gif" alt="{name}">'
  cards.append(f'<article><h2>{html.escape(c["kind"])} · {html.escape(c["rule"])}</h2><p>{html.escape(c["episode_id"])}</p>{media}</article>')
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Latent Evaluation · Process animations</title><style>body{font:16px/1.6 system-ui,sans-serif;margin:32px auto;max-width:1500px;padding:0 20px;color:#243247;background:#f5f7fa}h1{margin-bottom:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:24px}article{background:white;border:1px solid #dde2ea;border-radius:12px;padding:18px}h2{margin:0;font-size:20px}p{color:#546277}video,img{width:100%;border-radius:6px}a{color:#355b9b}</style><h1>GeoProcessBank · 同步过程动画</h1><p>同一物理轨迹：原始化学 + 六种表示。每种表示使用固定 PCA3；所有数值指标在原生空间计算。median、disagreement、random 均按预定义规则选择。</p><main>'+''.join(cards)+'</main></html>'
 (out/'index.html').write_text(page)

def render(cfg,records):
 from src.visualization.figures import all_figures
 from src.visualization.animations import render_animations
 from src.process_bank.ranges import audit_ranges,audit_semigroup_support,audit_order_coverage
 audit_ranges(records);audit_semigroup_support();audit_order_coverage();all_figures();render_animations(cfg);animation_gallery();write_report(cfg,records)
