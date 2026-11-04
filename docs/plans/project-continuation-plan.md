# Project continuation plan

## Objective
Develop two complementary, demonstrable engineering projects. Prioritize reliable, repeatable results and a useful product over resume claims. Preserve existing work and document both successes and failures. Do not edit any resume as part of this plan.

## Starting evidence
- The inference project is a fork; retain attribution. Its 8-worker least-loaded experiment reported 86.1 requests/s under latency/error criteria; the high-load point exceeded the CPU target. A separate five-minute hold sustained 45 requests/s. The failure experiment covered one worker kill and repeated inputs. See `docs/autotune-benchmark.md` and `README.md` in the inference project.
- LabFisika's cloud report recorded 50 profile reads/s at 81 ms p95 on a temporary 2-vCPU/2-GB copy for five minutes. That result is not a production ceiling or an AI capacity claim. See `docs/benchmarks/2026-10-24-capacity-release/README.md` in LabFisika. The report flags a Unicode note-sync mismatch; confirm whether it still exists before treating it as a current defect.
- These are historical findings, not proof of the current deployed version. Confirm repository state, deployed versions and available resources before acting.

## Phase 1: Baseline and research
1. Read applicable AGENTS.md files and project guides. Inspect git status and preserve unrelated changes. Identify the actual inference checkout, LabFisika checkout, existing tests and deployment instructions.
2. Reproduce the Unicode issue using synthetic accounts and representative Unicode strings; capture local edits, sync, reload, and exact code points. If fixed, document the evidence and choose the next verified reliability issue.
3. Research model-serving optimization using current primary sources: ONNX Runtime documentation, original research, and relevant project docs. Compare threading, batching, model selection, request routing, and hardware limits. Examine actual supported model inputs, accuracy evaluation, and deployment constraints. Cite sources and distinguish hypotheses from measurements.
4. Produce a short decision record ranking at least two inference directions by user benefit, feasibility, cost, safety, and measurable outcome. Prefer a bounded automatic configuration recommender only if evidence shows it improves on existing tooling. Retain a manual override and safe defaults.

## Phase 2: LabFisika reliability
- Trace and fix the verified root cause of the note-sync defect, preserving account isolation, existing notes and error recovery. Add meaningful regression coverage at the affected layers, including round-trip Unicode and conflicting edits where relevant.
- Verify local and realistic browser/API paths, with synthetic data. Run the project's required quality checks and isolated build, following its active-dev build rules.
- Prepare a deployable revision. Deploy only if the user has authorized the specific release; verify health, key user paths, and rollback readiness. Never overwrite production data with test data.
- Acceptance: the reproduced failure passes before/after checks, account isolation remains intact, the user sees save failures, and test/deployment limits are documented. If no current defect is reproduced, report that result and do not manufacture a fix.

## Phase 3: Inference improvement and evaluation
- Inspect existing gateway, worker, harness and tests before choosing design. Define workload, success criteria, and baseline before tuning. Compare against the same revision, models, dataset, machines, and measurement protocol; retain upstream attribution.
- Candidate direction A: a bounded offline autotuner that evaluates a small set of routing, batching and thread configurations on representative inputs, selects only configurations that meet explicit latency/error/resource constraints, and records model, data, machine and run metadata. Do not optimize against the final test set; hold out inputs/configuration validation where feasible. Surface accuracy checks and reject failures rather than silently selecting a faster incorrect model.
- Candidate direction B: failure recovery and long-running stability. Exercise worker loss, recovery, overload and restarts; measure error rate, recovery duration, p50/p95/p99 latency, memory and CPU. Avoid claiming availability from a single fault injection.
- Choose based on Phase 1 evidence. An optional longer run can follow a short calibration and must have a cost estimate, explicit resource ownership, hard stop conditions, cleanup, and a ledger. Do not create paid resources without authorization for the specific spend and test scope. No cloud testing on a live user service.
- Acceptance: build and targeted tests pass; identical-input correctness checks and load-generator limitations are reported; comparisons include run conditions, repetitions where practical, failed runs, and uncertainty. Record what is demonstrated versus untested. Results may be null or negative.
- Produce a runnable demo, reproducible instructions, concise architecture and results documentation, and a release/rollback plan where applicable. Do not imply the fork or upstream components were authored by the user.

## Phase 4: Review and delivery
Run independent verification against the actual diff, original requirements, and project instructions. Check deployment status, resource cleanup and costs. Present completed work, measured outcomes with scope, known limitations, next steps, and source links. Do not invent metrics or automatically update a resume. Commit/push only when authorized by the applicable project policy; deploy only with existing specific authorization.
