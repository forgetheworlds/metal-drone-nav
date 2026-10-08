"""Recompute the two matched actor-step endpoints from raw flight grades."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from navigation_scaling_results import read_grades, pool_grades

IDENTITY = ['env','scene_seed','start_x','start_y','start_z','start_yaw','goal_x','goal_y','goal_z','sensor_delay','command_delay']


def review(folder, phase):
    receipts = json.loads((folder / (phase + '-eval-receipts.json')).read_text())
    if len(receipts) != 92:
        raise ValueError('Incomplete paired evaluation matrix')
    rows = {}
    for receipt in receipts:
        key = (receipt['group'], receipt['seed'], receipt['panel'], receipt['profile'])
        if key in rows or receipt['exit'] != 0:
            raise ValueError('Duplicate or failed receipt')
        path = Path(receipt['csv'])
        # Bundles retain paths beneath their extraction root.
        relative = path.relative_to('/Users/muadhsambul/RL')
        local = folder.resolve().parents[1] / relative
        if local.exists():
            path = local
        rows[key] = read_grades(path)
    report = {}
    for panel, profile in sorted({(r['panel'], r['profile']) for r in receipts}):
        for seed in [1, 2]:
            for left, right in zip(rows[('control',seed,panel,profile)], rows[('treatment',seed,panel,profile)]):
                if any(left[key] != right[key] for key in IDENTITY):
                    raise ValueError('Paired task inputs differ')
        report[panel+'/'+profile] = {group: pool_grades([rows[(group,seed,panel,profile)] for seed in [1,2]]) for group in ['control','treatment']}
    return report


def plot(pilot, full, small, output):
    panels = [('open/nominal','Short open'), ('long-open/nominal','Long open'), ('clutter/nominal','Clutter'),
              ('composite/nominal','Course'), ('composite/both-delay','Both delays'), ('composite/combined','Combined stress')]
    figure, axes = plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    for axis, (key,title) in zip(axes.flat,panels):
        for group,color,label in [('control','#aa6565','Actor rate 1'),('treatment','#227b68','Actor rate 0.4')]:
            axis.plot([4.194304,41.94304],[pilot[key][group]['success']/256*100,full[key][group]['success']/256*100],marker='o',label=label,color=color)
        axis.axhline(small[key]['success']/256*100,color='#667085',linestyle='--',label='12k full-budget reference')
        axis.set(title=title,xlabel='Training transitions (millions)',ylabel='Stable success (%)',ylim=(0,100))
        axis.grid(alpha=.2);axis.legend(frameon=False,fontsize=8)
    figure.suptitle('Smaller actor updates help early, but full-budget retention remains incomplete')
    output.parent.mkdir(parents=True,exist_ok=True);figure.savefig(output,dpi=160);plt.close(figure)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--small-review',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('artifacts/plots/actor-step.png'))
    args=parser.parse_args();pilot=review(args.folder,'pilot');full=review(args.folder,'full')
    small=json.loads(args.small_review.read_text())['curves']['64']['41943040']
    result={'scope':'Paired source DEV endpoints, 4.19M and41.94M perseed; not blind FINAL or independent transfer.','pilot':pilot,'full':full,'small_final':small,'grade_rows_checked':23552}
    (args.folder/'paired-review.json').write_text(json.dumps(result,indent=2)+'\n');plot(pilot,full,small,args.out)
    print('Checked184paired panels/23552grades and exact task identities at both endpoints')
