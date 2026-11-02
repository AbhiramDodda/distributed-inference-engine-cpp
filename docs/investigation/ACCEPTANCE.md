# End-to-end acceptance and continuation

The owner wants a credible software engineering project, a revised software engineering resume, and a clean reviewable repository. Work through the plan continuously within the user's authorizations. Do not declare completion because code exists or a command returned zero. Complete every applicable gate below, or mark it **BLOCKED** with evidence, the precise missing permission or dependency, work already completed, and the next action. An unsupported claim or fabricated result is never an acceptable substitute.

## Required gates

| Gate | Evidence needed for PASS |
| --- | --- |
| Scope and state | Initial and final branch, revision and worktree state; preservation of existing changes; documented task and relevant constraints. |
| Investigation | Request path, measurement boundaries, competing hypotheses, supporting and opposing observations, and a justified selection recorded in FINDINGS. |
| Contribution | One focused implemented improvement, its rationale, actual source changes, and upstream attribution. If no change is evidence-supported, document that fact and mark this gate BLOCKED rather than labeling instrumentation alone an optimization. |
| Correctness | Relevant tests and numerical or behavioral comparison, exact commands, results, failures and skips. Response shape alone is not model accuracy. |
| Performance | Matched baseline and changed-source runs, provenance, workload and environment, raw local records, sample counts, latency, throughput, errors and variability. Report neutral or negative results faithfully. If cloud comparison is not safely available, use an appropriate local comparison when feasible and label its limits; otherwise BLOCKED. |
| Analysis | Interpretation tied to data, uncertainty, confounders, impact on the original hypothesis, reproducible commands and remaining risks. Do not represent local tests as AWS tests or single-host results as multi-host scaling. |
| AWS safety | If AWS was used: retained local logs, verified cleanup or explicit cleanup failure, and cost ledger updated after each operation. Stop further cloud runs on uncertain cleanup or budget. If no cloud action was needed or possible, state why and distinguish this from verified AWS work. |
| Portfolio | A concise existing README update or linked report with actual contribution, attribution, setup, tests, results and limitations. Avoid duplicate or promotional files. |
| Resume | Update only the authorized software-engineer resume family: add the verified project and retain only Kolosal.AI in Experience. Generate the one-page resume and certificate-appended copy; validate layout, extracted text and links per the resume skill and workspace rules. Report exact paths and limitations. |
| Repository review | Review tracked diff and every untracked file, check for sensitive or generated artifacts, preserve user work, run appropriate checks and report final status. No commit or push without authorization. The owner can review the change even when it remains uncommitted. |

## Final report

Run `python3 docs/investigation/audit.py` from the project root and include its output in the handoff. This read-only script checks only limited artifact existence and the resume page count if `pdfinfo` is available; it cannot prove that a project, benchmark, resume, or cleanup is correct. Do not treat its exit status as an acceptance result.

Require a separate final verification pass against the actual source diff, test output, raw comparison records, cost and cleanup evidence, and rendered resume. Prefer an independent reviewer or agent when available and authorized. Otherwise perform a clearly labeled self-review and report that independent review remains outstanding. Any failed gate returns the work to implementation or investigation; do not declare completion based on prose alone.

Give a gate-by-gate PASS/BLOCKED status with direct evidence and exact commands. Then summarize the change, why it was selected, before/after numbers and their limits, cost and cleanup, project and resume artifact paths, proposed resume bullets, and the Git state. Distinguish self-review from independent review. Do not say the project is production-ready or guarantee interview outcomes.

If the session may end before completion, update FINDINGS with what is done, exact remaining tasks, files and artifact paths, commands needed, current Git state, AWS resource and cleanup state, spend estimate, risks, and permissions needed. A later agent must be able to resume without repeating cloud operations or losing evidence. Do not mark a blocked task as done or run further billable work to hide a failure.

Persistence is not permission to bypass safety or spend limits. Recover from ordinary failures, try safe alternatives, and work on independent gates. If meaningful progress is impossible, pause with the precise blocker and resumption steps rather than retrying indefinitely or damaging data.
