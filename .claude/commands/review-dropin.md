---
description: Full 5-agent drop-in review of this package (fan out → aggregate → adversarially push back)
---

Run the saved multi-agent review workflow and report the result.

1. Invoke the Workflow tool with
   `{ scriptPath: ".claude/workflows/review-dropin.mjs" }` (it fans out 5 independent
   reviewers with identical input, aggregates + de-duplicates their findings, then runs one
   adversarial verifier per finding that re-checks it against the actual code and empirically
   tries the drop-in scenarios). To review a different repo, pass
   `{ scriptPath: ".claude/workflows/review-dropin.mjs", args: { repo: "<absolute path>" } }`.
   (Use `scriptPath`, not `name` — the `name` registry does not resolve files under
   `.claude/workflows/`.)
2. When it completes, read the full result file, then **independently verify the highest-severity
   and any drop-in-blocking findings yourself** (reproduce them) before trusting them.
3. Synthesize into a single prioritized report: the consensus drop-in verdict, each finding with
   its pushback verdict (confirmed / partially / refuted), corrected severity, and a recommended
   action. Call out where reviewers overstated and what the real blockers are.

Do not modify any code as part of this command — it is review-only. If the user wants fixes
afterward, propose a batch.
