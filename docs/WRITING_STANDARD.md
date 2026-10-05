# Writing standard

Use this standard for repository documents, especially the public README and project guides. Write for a technical reader who has not followed the work.

## Explain the work in reader order

Start with the real problem and what happens physically. Then say what this repository does and how it connects to the larger system. Explain implementation terms after the reader knows why they matter.

Use plain English and active verbs. Use first person for decisions and experiments the author actually made. Keep each paragraph focused, vary the rhythm, and remove repeated points, empty summaries, and promotional language. Keep the prose clean. Do not write to game an AI detector.

For this project, keep the system boundary clear: semantic task understanding and destination choice happen upstream; Metal-nav moves toward a supplied destination through geometry. Short-goal experiments describe their test setup, not the design's distance limit.

## State results plainly

Before publishing a number or technical detail, check it against current code, a report, a manifest, or raw evidence. Then write the finding directly. Include the policy, task set, denominator, and measure when the reader needs them to understand the comparison.

Describe what changed, what happened, and whether the change was kept. Report failures and tradeoffs in direct language. A failed experiment supports a conclusion about the tested setup; do not stretch it into a claim about every method or task.

Put the important scope beside the result, in one short sentence or its caption. Keep simulation, Webots, and physical flight distinct when the distinction matters. Explain whether a route is a geometric witness, whether a video is selected, and whether development tasks were used for selection. Avoid repeating the same caveat in later sections.

## Use visuals to explain

Use real plots, scene images, diagrams, and recordings that answer a reader's question. Do not invent data or imply a physical test that did not happen. Prefer a small diagram that explains one flow or system boundary.

Write literal captions. Name the policy and task when needed, state what the reader sees, and give the key number or selection status. Link important visuals to their source data or report.

## Order public documents for quick understanding

The first screen of a README should explain the project goal, this repository's role, the policy input and output, and how the component fits with the rest of the system. Show a useful visual early. Then cover behavior, results, failures, implementation choices, deployment, and reproduction.

Put dense implementation details lower in the document or in a focused report. Keep research logs and raw receipts intact. Make the evidence easier to find without erasing its history.

## Final pass

Before publishing, read the whole document once for facts and once for prose. Check key numbers and links. Make sure each caption describes its artifact. Remove stale claims, repeated caveats, generic conclusions, and unfinished placeholders. Leave the reader with a clear view of what works and what remains to learn.
