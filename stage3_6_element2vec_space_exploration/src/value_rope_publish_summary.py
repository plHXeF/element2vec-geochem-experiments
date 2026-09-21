"""Export only aggregate audit tables and raster figures to the public report."""
from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

from geochem_workspace import asset, output_path

import argparse
import shutil
from pathlib import Path

import pandas as pd


def export(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    results=source/'results'
    # Explicit allowlist: never copy per-sample rows, latent arrays or fitted bases.
    for name in ['quality','quality_modes','pca','pca_modes','quality_paired_bootstrap',
                 'paired_bootstrap','dropout_knn','final_probe_error_detail','full_data_numeric_summary',
                 'full_data_class_counts','pca_axis_associations','jacobian_spectra','rotary_frequency_energy']:
        shutil.copy2(results/f'{name}.csv',destination/f'{name}.csv')
    shutil.copy2(source/'final_probe_summary.csv',destination/'final_probe_summary.csv')
    shutil.copy2(source/'translation_anchor_control.csv',destination/'translation_anchor_control.csv')
    for name in ['full_data_summary','phase_kernel','training_integrity','final_verification']:
        shutil.copy2(results/f'{name}.json',destination/f'{name}.json')
    selections=[
        ('derivatives',['model','space','h'],['direction_cos','predictor_error','relative_curvature'],lambda d:d[d.natural.eq(True)]),
        ('paths',['model','space','family','scope','process'],['speed_cv','tangent_cos','turn_rate','uniformity_rmse','straightness'],lambda d:d),
        ('joints',['model','space','h'],['direction_cos','residual'],lambda d:d[d.natural.eq(True)]),
        ('linear_range',['model','space'],['linear_radius'],lambda d:d[d.natural.eq(True)]),
        ('controlled_backgrounds',['model','natural','relation'],['direction_cos'],lambda d:d),
        ('controlled_fine_steps',['model','h'],['fine_reference_cos','fine_reference_relative_error'],lambda d:d[d.natural.eq(True)]),
        ('context',['model','element','split'],['balanced_accuracy'],lambda d:d),
        ('shifts',['model','kind','h'],['normalized_sensitivity'],lambda d:d[d.clipped.eq(0)]),
        ('raw_ratio_shifts',['model','kind','h'],['normalized_sensitivity'],lambda d:d[d.clipped.eq(0)]),
        ('dropout',['model','rate'],['cosine','relative_displacement'],lambda d:d),
        ('collisions',['model','level'],['closest_far_over_local_step','far_pairs_below_one_local_step'],lambda d:d),
        ('support_collisions',['model','support'],['candidate_pairs','near_collision_pairs','min_ratio'],lambda d:d),
        ('original_differentiation_centroids',['model','space'],['speed_cv','tangent_cos','turn_rate','uniformity_rmse'],lambda d:d),
    ]
    for name,groups,metrics,select in selections:
        data=select(pd.read_csv(results/f'{name}.csv',low_memory=False))
        summary=data.groupby(groups,dropna=False)[metrics].agg(['count','mean','median','min','max'])
        summary.columns=['_'.join(c) for c in summary.columns]
        summary.reset_index().to_csv(destination/f'{name}_aggregate.csv',index=False)
    (destination/'figures').mkdir(exist_ok=True)
    for name in ['local_geometry','layer_geometry','quality_guardrails','global_pca','training_curves','phase_collision']:
        shutil.copy2(source/'figures'/f'{name}.png',destination/'figures'/f'{name}.png')
    print(f'Exported aggregate tables and raster figures to {destination}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=output_path("audit"))
    p.add_argument('--report-dir',type=Path,default=output_path("reports"))
    args=_guard_args(p.parse_args());export(args.output,args.report_dir)
    _guard_args(args)


if __name__=='__main__':main()
