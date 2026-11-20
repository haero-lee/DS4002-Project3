"""Compare saved U-Net Dice scores with the published UNet 3+ benchmark.

This analysis reads the existing per-mouse CSV, split, config, and published
benchmark. It does not train, run inference, or recalculate Dice from masks.
Run from the repository root:
    python SCRIPTS/05_analyze_results.py --run_dir OUTPUT/UNet2D_2026-11-16_17-16-24

Requires Python 3.9+ and matplotlib for the two figures. Use --no-figures for
CSV/text outputs with the standard library only. Outputs go into the run's
analysis/ folder unless --out_dir is supplied. Original artifacts are unchanged.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ('diastolic_blood_pool', 'blood_pool', 'ED', 'Blood pool, ED'),
    ('diastolic_myocardium', 'myocardium', 'ED', 'Myocardium, ED'),
    ('systolic_blood_pool', 'blood_pool', 'ES', 'Blood pool, ES'),
    ('systolic_myocardium', 'myocardium', 'ES', 'Myocardium, ES'),
    ('average', 'overall', 'ED_and_ES', 'Overall'),
]
CORE = [m[0] for m in METRICS[:-1]]
PAPER_URL = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13237537/#tbl1'


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def score(value, description):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f'{description}: expected a finite Dice value in [0, 1]')
    return number


def load_inputs(run_dir, benchmark_path):
    """Require complete finite scores and exact agreement with saved test IDs."""
    split = json.loads((run_dir / 'split.json').read_text())
    config = json.loads((run_dir / 'config.json').read_text())
    seen = set()
    for group in ('train', 'val', 'test'):
        mice = split[group]
        if not mice or len(mice) != len(set(mice)) or seen.intersection(mice):
            raise ValueError(f'Duplicate, overlapping, or empty {group} split')
        if set(mice) != set(config[f'{group}_mice']):
            raise ValueError(f'Split/config disagreement in {group} mice')
        seen.update(mice)

    rows = read_csv(run_dir / 'eval/test_dice_per_mouse.csv')
    ids = [r['mouse'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(split['test']):
        raise ValueError('Score rows must contain every saved test mouse exactly once')
    for row in rows:
        for key, _, _, _ in METRICS:
            row[key] = score(row[key], f"{row['mouse']} {key}")
        if not math.isclose(row['average'], statistics.mean(row[k] for k in CORE),
                            rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{row['mouse']}: overall is not the four-score mean")
    if len(rows) < 2:
        raise ValueError('At least two test mice are needed for a sample SD')

    published = {}
    for row in read_csv(benchmark_path):
        key = (row['structure'], row['phase'])
        if key in published:
            raise ValueError(f'Duplicate benchmark metric: {key}')
        if (row['result_origin'] != 'published_not_reproduced'
                or row['cohort'] != 'internal_9.4T' or int(row['test_mice']) != 25
                or row['doi'] != '10.1016/j.jocmr.2026.102706'
                or row['table_reference'] != 'Table 1'):
            raise ValueError('Expected the published internal 9.4T Table 1 cohort')
        row['dice_mean'] = score(row['dice_mean'], f'{key} published mean')
        row['dice_sd'] = score(row['dice_sd'], f'{key} published SD')
        published[key] = row
    if set(published) != {(m[1], m[2]) for m in METRICS}:
        raise ValueError('Expected exactly five published benchmark metrics')
    return rows, split, config, published


def build_comparison(rows, published):
    """Recompute means/sample SDs from mice; differences use published rounding."""
    result = []
    for key, structure, phase, label in METRICS:
        values = [r[key] for r in rows]
        paper = published[(structure, phase)]
        mean = statistics.mean(values)
        result.append(dict(
            metric=label, structure=structure, phase=phase,
            unet2d_origin='measured_saved_per_mouse_scores', unet2d_n=len(rows),
            unet2d_mean=mean, unet2d_sd=statistics.stdev(values),
            unet3plus_origin=paper['result_origin'], unet3plus_n=int(paper['test_mice']),
            unet3plus_mean=paper['dice_mean'], unet3plus_sd=paper['dice_sd'],
            difference_ours_minus_published=mean-paper['dice_mean'],
            comparison_type='descriptive_only', doi=paper['doi'],
            table_reference=paper['table_reference'], source_url=paper['source_url'],
        ))
    return result


def audit_context(run_dir, rows, split, config):
    """Inspect saved context without loading model weights or MRI arrays."""
    notes = []
    counts_path = ROOT / 'OUTPUT/eda_outputs/14_step8_labeled_counts.csv'
    if counts_path.exists():
        extras = [(r['file'], int(r['labeled_frames'])) for r in read_csv(counts_path)
                  if r['file'].removesuffix('.h5') in split['test']
                  and int(r['labeled_frames']) > 2]
        if extras:
            notes.append('EDA lists additional annotated frames in test mice: '
                         + ', '.join(f'{name} ({n} frames)' for name, n in extras)
                         + '. This needs phase-selection review; it is not evidence of leakage.')
    notes.extend([
        f"Saved config test_n={config['test_n']} counts extracted labeled 2D inputs, "
        'not the number of ED/ES slices actually scored. The paper reports 460 ED/ES test images.',
        'The saved split matches the released notebook lists checked in unet2d_utils.py; '
        'identity with the final paper training partition is not independently confirmed. '
        'The paper says additional inflow frames were placed in training.',
        'Uploaded code infers ED as the earliest annotated frame and ES as the remaining '
        'frame with minimum guarded blood-pool volume. Actual selected indices were not saved.',
        'Uploaded evaluation pools all slices separately for each class at ED and ES, '
        'then averages the four Dice values per mouse. It does not pool ED and ES pixel '
        'counts together. Every mouse receives equal weight; SD here uses ddof=1.',
        'The evaluator treats an empty class in both masks as Dice 1 and uses smoothing '
        '1e-6 otherwise. Post-processing is enabled by default in the training script, '
        'but the on/off flag is not stored in config.json. Its island-removal code can '
        'retain multiple qualifying components; exact equivalence with the paper is not established.',
        'No source MRI files or saved prediction masks are present here. This analysis '
        'checks score/split consistency, not pixel alignment, phase correctness, or mask-level Dice.',
        'Published results are rounded to two decimals and their overall aggregation '
        'is not fully specified. Differences are descriptive, not evidence of architectural '
        'superiority or equivalence. No between-model significance test is possible from summaries alone.',
    ])
    # Verify that re-aggregation agrees with the saved rounded training report.
    summary = (run_dir / 'eval/test_summary.txt').read_text()
    for key, _, _, _ in METRICS:
        match = re.search(rf'^{key}\s+([0-9.]+)\s+±\s+([0-9.]+)', summary, re.M)
        values = [r[key] for r in rows]
        if match is None or any(abs(a-b) > 0.000051 for a, b in zip(
                (statistics.mean(values), statistics.stdev(values)), map(float, match.groups()))):
            raise ValueError(f'Saved test summary disagrees with per-mouse scores: {key}')
    return notes



def make_figures(comparison, out_dir):
    """Plot the same five CSV rows; SD bars describe variation, not uncertainty."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})
    positions = list(range(len(comparison)))
    labels = [r['metric'] for r in comparison]
    n = comparison[0]['unet2d_n']

    # A full 0–1 scale and printed values make the measured scores easy to read.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=.20, right=.76, top=.84, bottom=.19)
    colors = ['#3478AD'] * (len(comparison)-1) + ['#194466']
    ax.barh(positions, [r['unet2d_mean'] for r in comparison], height=.52,
            color=colors, xerr=[r['unet2d_sd'] for r in comparison],
            error_kw={'ecolor': '#253746', 'capsize': 4, 'elinewidth': 1.5})
    ax.set(yticks=positions, yticklabels=labels, xlim=(0, 1),
           xlabel='Dice coefficient (0–1; higher is better)')
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.18)
    for i, row in enumerate(comparison):
        ax.text(1.035, i, f"{row['unet2d_mean']:.3f} ± {row['unet2d_sd']:.3f}",
                transform=ax.get_yaxis_transform(), va='center', color=colors[i],
                fontweight='bold' if row['metric'] == 'Overall' else 'normal')
    fig.text(.20, .93, '2D U-Net: measured test results', fontsize=17, weight='bold')
    fig.text(.20, .875, f'{n} held-out mice · mean ± sample standard deviation', fontsize=11)
    fig.text(.02, .055,
             'ED = end-diastole; ES = end-systole. Error bars show standard deviations across mice.\n'
             'Overall averages the four class/phase scores per mouse, then weights mice equally.',
             fontsize=10)
    fig.savefig(out_dir/'fig_unet2d_results.png', dpi=200)
    plt.close(fig)

    # Show both means/SDs and the signed gap, without implying a paired test.
    fig, ax = plt.subplots(figsize=(12, 6.3))
    fig.subplots_adjust(left=.19, right=.79, top=.83, bottom=.25)
    ax.errorbar([r['unet2d_mean'] for r in comparison], [i-.13 for i in positions],
                xerr=[r['unet2d_sd'] for r in comparison], fmt='o', color='#2463A0',
                capsize=4, markersize=7, label=f'2D U-Net — measured (n={n})')
    paper_n = comparison[0]['unet3plus_n']
    ax.errorbar([r['unet3plus_mean'] for r in comparison], [i+.13 for i in positions],
                xerr=[r['unet3plus_sd'] for r in comparison], fmt='s', color='#BD5A22',
                capsize=4, markersize=7, label=f'UNet 3+ — published (n={paper_n})')
    ax.set(yticks=positions, yticklabels=labels, xlim=(.75, 1),
           xlabel='Dice coefficient (detail view: 0.75–1.00)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.18)
    ax.text(1.045, 1.045, 'Mean difference', transform=ax.transAxes, fontsize=10, weight='bold')
    for i, row in enumerate(comparison):
        ax.text(1.045, i, f"{row['difference_ours_minus_published']:+.3f}",
                transform=ax.get_yaxis_transform(), va='center')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.16), frameon=False, ncol=2, fontsize=10)
    fig.text(.19, .94, '2D U-Net vs. published UNet 3+', fontsize=17, weight='bold')
    fig.text(.19, .88, 'Internal 9.4T benchmark · means with standard deviation bars', fontsize=11)
    fig.text(.02, .045,
             'Difference = our mean − published mean; negative values indicate a lower score. '
             'Published source: Shah et al., Table 1.\n'
             'Descriptive comparison only. Protocol equivalence is unconfirmed; '
             'SD bars are not confidence intervals.', fontsize=9.5)
    fig.savefig(out_dir/'fig_dice_comparison.png', dpi=200)
    plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, default=ROOT/'DATA/published_unet3plus.csv')
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--no-figures', action='store_true',
                        help='Write CSV/text/manifest only; matplotlib is not required')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir/'analysis'
    rows, split, config, published = load_inputs(run_dir, args.benchmark)
    comparison = build_comparison(rows, published)
    notes = audit_context(run_dir, rows, split, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_version = None if args.no_figures else make_figures(comparison, out_dir)
    with (out_dir/'comparison_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparison)

    overall = comparison[-1]
    lines = [
        '2D U-Net vs published UNet 3+: descriptive comparison',
        f'Run: {run_dir.name}; n={len(rows)} test mice; all mice retained.',
        'Metric | U-Net mean ± sample SD | Published mean ± SD | Ours minus published',
    ]
    for r in comparison:
        lines.append(f"{r['metric']} | {r['unet2d_mean']:.4f} ± {r['unet2d_sd']:.4f} | "
                     f"{r['unet3plus_mean']:.2f} ± {r['unet3plus_sd']:.2f} | "
                     f"{r['difference_ours_minus_published']:+.4f}")
    lines += [
        '', f"Target mean Dice >=0.90: {'MET' if overall['unet2d_mean'] >= .9 else 'NOT MET'}.",
        '', 'The U-Net point estimates are lower in all five rows. Gaps are small at ED '
        'and larger at ES, especially myocardium. This does not establish statistical '
        'equivalence at ED or a causal effect of model architecture.',
        '', 'Method and limitations:', *['- '+note for note in notes],
        '', 'Remaining work: verify actual ED/ES indices (particularly Mouse91 and Mouse125), '
        'confirm post-processing settings, and inspect real best/worst prediction overlays. '
        'No overlays or slice-position Dice can be reconstructed from the score CSV.',
        '', 'Published source: '+PAPER_URL,
    ]
    report = '\n'.join(lines)+'\n'
    (out_dir/'analysis_summary.txt').write_text(report)

    inputs = [run_dir/'eval/test_dice_per_mouse.csv', run_dir/'split.json',
              run_dir/'config.json', run_dir/'eval/test_summary.txt',
              args.benchmark, Path(__file__), ROOT/'SCRIPTS/02_utility.py',
              ROOT/'SCRIPTS/03_train_unet2d.py', ROOT/'SCRIPTS/04_unet2d_utils.py']
    counts = ROOT/'OUTPUT/eda_outputs/14_step8_labeled_counts.csv'
    if counts.exists():
        inputs.append(counts)
    checkpoint = run_dir/'best_model.keras'
    if checkpoint.exists():
        inputs.append(checkpoint)
    def relative(path):
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path.resolve())
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(
        repository_head=commit, source_files_sha256={relative(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in inputs},
        python=platform.python_version(),
        dependencies={'matplotlib': matplotlib_version} if matplotlib_version else {},
        figures_generated=not args.no_figures,
        analysis='Re-aggregation of uploaded scores; no mask-level reevaluation or training',
        data_source_commit='Not recorded in uploaded training config',
        checkpoint_identity='Uploaded best_model.keras hash; link to original evaluation not independently verified',
    )
    (out_dir/'analysis_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(report)
    print(f'Comparison outputs saved in: {out_dir}')


if __name__ == '__main__':
    main()
