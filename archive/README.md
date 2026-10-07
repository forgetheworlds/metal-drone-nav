# Preserved unpublished experiments

`unpublished-experiments.tar.gz` stores earlier local drafts, unused experiment
source and rejected media byte for byte. `unpublished-manifest.json` records
original paths, sizes and SHA-256 values. The archive includes its own manifest.
Original files were removed from the working tree only after hash verification.
No tracked source, selected policy, live input or native recording was moved.

These drafts are not active implementations or verified results. Some reports
reflect superseded task definitions or incomplete audits. The doorway and
connected-room MP4s here are rejected reconstructions; they are not native
Webots footage and must not enter the results gallery as flight evidence.

Inspect before restoring. Extract into a temporary directory, compare the
manifest, then restore only the files needed for a specific reviewed experiment.
Keep the experiment's old code/data contract; do not resume failed jobs merely
because their source is available. Local `results/` and published evidence
archives retain the associated outcomes and current verified work.
