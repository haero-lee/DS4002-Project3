"""Compare saved U-Net Dice scores with the published UNet 3+ benchmark.

===============================================================================
05_analyze_results.py
===============================================================================
PURPOSE
    Final comparison step of DS 4002 Project 3 (2D U-Net vs. UNet 3+ for mouse
    LV segmentation). It takes the per-mouse test Dice scores saved by
    03_train_unet2d.py, checks them for consistency, recomputes the summary
    statistics, and compares them with the published UNet 3+ results
    (Shah et al., 2026, Table 1, internal 9.4T test cohort of 25 mice).

This analysis reads the existing per-mouse CSV, split, config, and published
benchmark. It does not train, run inference, or recalculate Dice from masks.
Run from the repository root:
    python SCRIPTS/05_analyze_results.py --run_dir OUTPUT/UNet2D_2026-11-16_17-16-24

Requires Python 3.9+ and matplotlib for the two figures. Use --no-figures for
CSV/text outputs with the standard library only. Outputs go into the run's
analysis/ folder unless --out_dir is supplied. Original artifacts are unchanged.

INPUTS (read only, never modified)
    <run_dir>/eval/test_dice_per_mouse.csv   per-mouse Dice from training
    <run_dir>/eval/test_summary.txt          rounded summary from training
    <run_dir>/split.json, <run_dir>/config.json   mice in each set, settings
    DATA/published_unet3plus.csv             published UNet 3+ benchmark
    OUTPUT/eda_outputs/14_step8_labeled_counts.csv   (optional) EDA frame counts

OUTPUTS  ->  <run_dir>/analysis/  (or --out_dir)
    comparison_table.csv      our mean ± SD vs. published mean ± SD, per metric
    analysis_summary.txt      plain-text results and method/limitation notes
    fig_unet2d_results.png    our five Dice means on a 0-1 scale
    fig_dice_comparison.png   our vs. published means, with the signed gap
    analysis_manifest.json    SHA-256 hashes of every input, Python version, git commit

OPTIONS
    --run_dir     training run folder (required)
    --benchmark   published benchmark CSV (default DATA/published_unet3plus.csv)
    --out_dir     where to write outputs (default <run_dir>/analysis)
    --no-figures  skip the figures (no matplotlib needed)

NOTES
    - The script stops with an error if any check fails (missing or duplicate
      mice, overlapping splits, scores outside [0, 1], a mismatch with the
      saved summary, or an unexpected benchmark file). This guards against
      silently comparing the wrong files.
    - The comparison is descriptive: the published values are summary
      statistics, so no paired significance test between models is possible.
===============================================================================
"""

# ----------------------------------------------------------------------------
# Imports (all from the Python standard library; matplotlib is imported later,
# only if figures are requested)
# ----------------------------------------------------------------------------
import argparse      # read command-line options
import csv           # read and write CSV tables
import hashlib       # SHA-256 fingerprints of input files for the manifest
import json          # read split.json / config.json, write the manifest
import math          # isfinite / isclose checks on scores
from pathlib import Path   # file paths that work on any operating system
import platform      # record the Python version in the manifest
import re            # find numbers in the saved text summary
import statistics    # mean and sample standard deviation
import subprocess    # ask git for the current commit (for the manifest)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
# Repository root = the folder above SCRIPTS/ (lets default paths work from anywhere)
ROOT = Path(__file__).resolve().parents[1]
# The five metrics compared: (column in our CSV, structure name in the benchmark
# CSV, cardiac phase in the benchmark CSV, label used in tables and figures)
METRICS = [
    ('diastolic_blood_pool', 'blood_pool', 'ED', 'Blood pool, ED'),
    ('diastolic_myocardium', 'myocardium', 'ED', 'Myocardium, ED'),
    ('systolic_blood_pool', 'blood_pool', 'ES', 'Blood pool, ES'),
    ('systolic_myocardium', 'myocardium', 'ES', 'Myocardium, ES'),
    ('average', 'overall', 'ED_and_ES', 'Overall'),
]
# The four class/phase scores (everything except "average")
CORE = [m[0] for m in METRICS[:-1]]
# Link to the published table, printed in the summary
PAPER_URL = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13237537/#tbl1'


def read_csv(path):
    """Read a CSV file into a list of dictionaries (one per row).
    'utf-8-sig' also handles files saved by Excel with a byte-order mark."""
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def score(value, description):
    """Convert a text value to a number and check it is a valid Dice (0 to 1)."""
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f'{description}: expected a finite Dice value in [0, 1]')
    return number


def load_inputs(run_dir, benchmark_path):
    """Require complete finite scores and exact agreement with saved test IDs."""
    # ---- Split and config: each set non-empty, no duplicates, no overlap ----
    split = json.loads((run_dir / 'split.json').read_text())
    config = json.loads((run_dir / 'config.json').read_text())
    seen = set()   # mice already assigned to an earlier set
    for group in ('train', 'val', 'test'):
        mice = split[group]
        # Fail if the set is empty, has a repeated mouse, or shares a mouse with
        # an earlier set (that would be data leakage)
        if not mice or len(mice) != len(set(mice)) or seen.intersection(mice):
            raise ValueError(f'Duplicate, overlapping, or empty {group} split')
        # split.json and config.json were written separately; they must agree
        if set(mice) != set(config[f'{group}_mice']):
            raise ValueError(f'Split/config disagreement in {group} mice')
        seen.update(mice)

    # ---- Per-mouse test scores: exactly the saved test mice, valid values ----
    rows = read_csv(run_dir / 'eval/test_dice_per_mouse.csv')
    ids = [r['mouse'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(split['test']):
        raise ValueError('Score rows must contain every saved test mouse exactly once')
    for row in rows:
        # Convert each score to a number and check it lies in [0, 1]
        for key, _, _, _ in METRICS:
            row[key] = score(row[key], f"{row['mouse']} {key}")
        # "average" must equal the mean of the four class/phase scores
        if not math.isclose(row['average'], statistics.mean(row[k] for k in CORE),
                            rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{row['mouse']}: overall is not the four-score mean")
    # A sample standard deviation needs at least two values
    if len(rows) < 2:
        raise ValueError('At least two test mice are needed for a sample SD')

    # ---- Published benchmark: exactly the five expected Table 1 rows ----------
    published = {}
    for row in read_csv(benchmark_path):
        key = (row['structure'], row['phase'])
        if key in published:
            raise ValueError(f'Duplicate benchmark metric: {key}')
        # Make sure every row is the published internal 9.4T cohort of 25 mice
        # from Table 1 of the right paper (not some other cohort or source)
        if (row['result_origin'] != 'published_not_reproduced'
                or row['cohort'] != 'internal_9.4T' or int(row['test_mice']) != 25
                or row['doi'] != '10.1016/j.jocmr.2026.102706'
                or row['table_reference'] != 'Table 1'):
            raise ValueError('Expected the published internal 9.4T Table 1 cohort')
        row['dice_mean'] = score(row['dice_mean'], f'{key} published mean')
        row['dice_sd'] = score(row['dice_sd'], f'{key} published SD')
        published[key] = row
    # The benchmark must cover exactly our five metrics
    if set(published) != {(m[1], m[2]) for m in METRICS}:
        raise ValueError('Expected exactly five published benchmark metrics')
    return rows, split, config, published


def build_comparison(rows, published):
    """Recompute means/sample SDs from mice; differences use published rounding."""
    result = []
    for key, structure, phase, label in METRICS:
        values = [r[key] for r in rows]          # this metric for every test mouse
        paper = published[(structure, phase)]    # matching published row
        mean = statistics.mean(values)           # every mouse weighted equally
        # One row of comparison_table.csv; "origin" columns record where each
        # number came from (our measurement vs. the published paper)
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
    notes = []   # method and limitation notes for analysis_summary.txt

    # Flag test mice that the EDA found have more than two labeled frames
    # (i.e. extra "inflow" frames), since they affect which frame is chosen as ES
    counts_path = ROOT / 'OUTPUT/eda_outputs/14_step8_labeled_counts.csv'
    if counts_path.exists():
        extras = [(r['file'], int(r['labeled_frames'])) for r in read_csv(counts_path)
                  if r['file'].removesuffix('.h5') in split['test']
                  and int(r['labeled_frames']) > 2]
        if extras:
            notes.append('EDA lists additional annotated frames in test mice: '
                         + ', '.join(f'{name} ({n} frames)' for name, n in extras)
                         + '. This needs phase-selection review; it is not evidence of leakage.')

    # Fixed notes describing how the scores were produced and their limits
    notes.extend([
        f"Saved config test_n={config['test_n']} counts extracted labeled 2D inputs, "
        'not the number of ED/ES slices actually scored. The paper reports 460 ED/ES test images.',
        'The saved split matches the released notebook lists checked in 04_unet2d_utils.py; '
        'identity with the final paper training partition is not independently confirmed. '
        'The paper says additional inflow frames were placed in training.',
        'Uploaded code infers ED as the earliest annotated frame and ES as the remaining '
        'frame with minimum guarded blood-pool volume. Training did not save the indices; '
        'the later test_segmentations/test_segmentation_summary.csv records the frames '
        'selected by the same rule (1-based), when that artifact is present.',
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
    # For each metric, read "mean ± SD" from test_summary.txt and require our
    # recomputed values to match it to within rounding (4 decimal places)
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
    # Imported here so --no-figures works without matplotlib installed
    import matplotlib
    matplotlib.use('Agg')            # draw to files; no screen needed
    import matplotlib.pyplot as plt

    # Shared style: readable font size, no top/right borders
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})
    positions = list(range(len(comparison)))   # one row per metric
    labels = [r['metric'] for r in comparison]
    n = comparison[0]['unet2d_n']              # number of test mice

    # ---- Figure 1: our results only (horizontal bars, mean ± SD) ------------
    # A full 0–1 scale and printed values make the measured scores easy to read.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=.20, right=.76, top=.84, bottom=.19)
    # Same blue for the four class/phase metrics; darker blue for "Overall"
    colors = ['#3478AD'] * (len(comparison)-1) + ['#194466']
    ax.barh(positions, [r['unet2d_mean'] for r in comparison], height=.52,
            color=colors, xerr=[r['unet2d_sd'] for r in comparison],
            error_kw={'ecolor': '#253746', 'capsize': 4, 'elinewidth': 1.5})
    ax.set(yticks=positions, yticklabels=labels, xlim=(0, 1),
           xlabel='Dice coefficient (0–1; higher is better)')
    ax.invert_yaxis()                # first metric at the top
    ax.set_axisbelow(True)           # grid behind the bars
    ax.grid(axis='x', alpha=.18)
    # Print "mean ± SD" to the right of each bar
    for i, row in enumerate(comparison):
        ax.text(1.035, i, f"{row['unet2d_mean']:.3f} ± {row['unet2d_sd']:.3f}",
                transform=ax.get_yaxis_transform(), va='center', color=colors[i],
                fontweight='bold' if row['metric'] == 'Overall' else 'normal')
    # Title, subtitle and footnote
    fig.text(.20, .93, '2D U-Net: measured test results', fontsize=17, weight='bold')
    fig.text(.20, .875, f'{n} held-out mice · mean ± sample standard deviation', fontsize=11)
    fig.text(.02, .055,
             'ED = end-diastole; ES = end-systole. Error bars show standard deviations across mice.\n'
             'Overall averages the four class/phase scores per mouse, then weights mice equally.',
             fontsize=10)
    fig.savefig(out_dir/'fig_unet2d_results.png', dpi=200)
    plt.close(fig)

    # ---- Figure 2: ours vs. published (dots with SD bars + signed gap) -------
    # Show both means/SDs and the signed gap, without implying a paired test.
    fig, ax = plt.subplots(figsize=(12, 6.3))
    fig.subplots_adjust(left=.19, right=.79, top=.83, bottom=.25)
    # Our model: blue circles, shifted slightly up within each row
    ax.errorbar([r['unet2d_mean'] for r in comparison], [i-.13 for i in positions],
                xerr=[r['unet2d_sd'] for r in comparison], fmt='o', color='#2463A0',
                capsize=4, markersize=7, label=f'2D U-Net — measured (n={n})')
    # Published UNet 3+: orange squares, shifted slightly down within each row
    paper_n = comparison[0]['unet3plus_n']
    ax.errorbar([r['unet3plus_mean'] for r in comparison], [i+.13 for i in positions],
                xerr=[r['unet3plus_sd'] for r in comparison], fmt='s', color='#BD5A22',
                capsize=4, markersize=7, label=f'UNet 3+ — published (n={paper_n})')
    # Zoomed x-axis (0.75-1.00) so the small differences are visible
    ax.set(yticks=positions, yticklabels=labels, xlim=(.75, 1),
           xlabel='Dice coefficient (detail view: 0.75–1.00)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.18)
    # Right-hand column: our mean minus the published mean for each metric
    ax.text(1.045, 1.045, 'Mean difference', transform=ax.transAxes, fontsize=10, weight='bold')
    for i, row in enumerate(comparison):
        ax.text(1.045, i, f"{row['difference_ours_minus_published']:+.3f}",
                transform=ax.get_yaxis_transform(), va='center')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.16), frameon=False, ncol=2, fontsize=10)
    # Title, subtitle and footnote
    fig.text(.19, .94, '2D U-Net vs. published UNet 3+', fontsize=17, weight='bold')
    fig.text(.19, .88, 'Internal 9.4T benchmark · means with standard deviation bars', fontsize=11)
    fig.text(.02, .045,
             'Difference = our mean − published mean; negative values indicate a lower score. '
             'Published source: Shah et al., Table 1.\n'
             'Descriptive comparison only. Protocol equivalence is unconfirmed; '
             'SD bars are not confidence intervals.', fontsize=9.5)
    fig.savefig(out_dir/'fig_dice_comparison.png', dpi=200)
    plt.close(fig)
    # Returned so the manifest can record which matplotlib version drew the figures
    return matplotlib.__version__


def main():
    # ---- 1. Command-line options -------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, default=ROOT/'DATA/published_unet3plus.csv')
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--no-figures', action='store_true',
                        help='Write CSV/text/manifest only; matplotlib is not required')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    # Default output folder is <run_dir>/analysis
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir/'analysis'

    # ---- 2. Load and check inputs, compute the comparison ---------------------
    rows, split, config, published = load_inputs(run_dir, args.benchmark)
    comparison = build_comparison(rows, published)
    notes = audit_context(run_dir, rows, split, config)

    # ---- 3. Figures and comparison table -------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_version = None if args.no_figures else make_figures(comparison, out_dir)
    with (out_dir/'comparison_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparison)

    # ---- 4. Plain-text summary (analysis_summary.txt) -------------------------
    overall = comparison[-1]   # the "Overall" row
    lines = [
        '2D U-Net vs published UNet 3+: descriptive comparison',
        f'Run: {run_dir.name}; n={len(rows)} test mice; all mice retained.',
        'Metric | U-Net mean ± sample SD | Published mean ± SD | Ours minus published',
    ]
    # One line per metric: ours (4 decimals) vs. published (2 decimals, as reported)
    for r in comparison:
        lines.append(f"{r['metric']} | {r['unet2d_mean']:.4f} ± {r['unet2d_sd']:.4f} | "
                     f"{r['unet3plus_mean']:.2f} ± {r['unet3plus_sd']:.2f} | "
                     f"{r['difference_ours_minus_published']:+.4f}")
    # Finish-line check, interpretation, method notes, remaining work, source
    lines += [
        '', f"Target mean Dice >=0.90: {'MET' if overall['unet2d_mean'] >= .9 else 'NOT MET'}.",
        '', 'The U-Net point estimates are lower in all five rows. Gaps are small at ED '
        'and larger at ES, especially myocardium. This does not establish statistical '
        'equivalence at ED or a causal effect of model architecture.',
        '', 'Method and limitations:', *['- '+note for note in notes],
        '', 'Unresolved limitations: selected phases (especially Mouse91 and Mouse125) '
        'are inferred rather than independently expert-verified; the original config omits '
        'the post-processing flag and dataset revision. Basal/middle/apical Dice promised '
        'in MI2 was not completed: the saved score CSV contains whole-stack metrics only. '
        'The README records this deviation; EDA area profiles and mid-ventricular '
        'overlays do not establish accuracy by slice position.',
        '', 'Published source: '+PAPER_URL,
    ]
    report = '\n'.join(lines)+'\n'
    (out_dir/'analysis_summary.txt').write_text(report)

    # ---- 5. Manifest: fingerprints of every input, for reproducibility --------
    # Files whose exact contents produced these results
    inputs = [run_dir/'eval/test_dice_per_mouse.csv', run_dir/'split.json',
              run_dir/'config.json', run_dir/'eval/test_summary.txt',
              args.benchmark, Path(__file__), ROOT/'SCRIPTS/02_utility.py',
              ROOT/'SCRIPTS/03_train_unet2d.py', ROOT/'SCRIPTS/04_unet2d_utils.py']
    # Optional files, included only if present
    counts = ROOT/'OUTPUT/eda_outputs/14_step8_labeled_counts.csv'
    if counts.exists():
        inputs.append(counts)
    checkpoint = run_dir/'best_model.keras'
    if checkpoint.exists():
        inputs.append(checkpoint)

    def relative(path):
        """Show paths relative to the repository root when possible."""
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path.resolve())
    # Current git commit of this repository (None if git is unavailable)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    # SHA-256 hash of each input: any change to a file changes its hash
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

    # ---- 6. Print the summary to the screen ---------------------------------------
    print(report)
    print(f'Comparison outputs saved in: {out_dir}')


# Run main() only when this file is executed directly
if __name__ == '__main__':
    main()
