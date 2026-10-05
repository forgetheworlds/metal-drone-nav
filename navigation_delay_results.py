"""Plot the complete delay-learning comparison from its hashed evidence archive."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from navigation_delay_review import review_archive


def plot(archive, output):
    report = review_archive(archive)
    rows = report['results']
    labels = ['Nominal', 'Sensor delay', 'Command delay', 'Both delays', 'Combined stress']
    keys = ['nominal', 'sensor-delay', 'command-delay', 'both-delay', 'combined']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for arm, label, color in [('C', 'Nominal training', '#6e7f90'), ('T', 'Mixed delay training', '#227b68')]:
        success = [100*rows['composite/'+key][arm]['success']/256 for key in keys]
        contacts = [rows['composite/'+key][arm]['contacts'] for key in keys]
        axes[0].plot(labels, success, marker='o', color=color, label=label)
        axes[1].plot(labels, contacts, marker='o', color=color, label=label)
    axes[0].set_ylabel('Stable success (%)');axes[0].set_ylim(0, 100)
    axes[1].set_ylabel('Contacts / 256 policy-task trials');axes[1].set_ylim(0, 80)
    for axis in axes:
        axis.grid(alpha=.2);axis.tick_params(axis='x', rotation=20);axis.legend(frameon=False)
    fig.suptitle('Frozen combined-course evaluation after matched PPO training')
    fig.text(.5, -.04, 'Two trained seeds on the same 128 development tasks; ideal ego, zero wind. No policy selection from stress results.', ha='center', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches='tight');plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, default=Path('artifacts/plots/delay-learning.png'))
    args = parser.parse_args();plot(args.archive,args.out)
