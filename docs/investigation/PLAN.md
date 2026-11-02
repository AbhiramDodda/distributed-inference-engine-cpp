# Performance and reliability investigation

## Objective and boundary

Investigate the existing CPU inference system, select and implement one evidence-supported performance improvement, verify correctness, compare results, analyze the outcome, and prepare a portfolio-ready handoff. Investigate performance first, then correctness, reliability, security, and operations where they affect an honest benchmark or safe operation. This is a two-day project, not an audit of every possible defect. Timebox exploratory work; record lower-priority leads instead of expanding scope.

Portfolio goal: make the owner's distinct engineering contribution easy to inspect and discuss. The intended outcome is an explained problem, one justified change if evidence supports it, meaningful correctness checks, a fair comparison, and clear documentation. A measured regression or inconclusive result is still useful engineering evidence; do not manufacture a win or claim that an inherited feature was authored by the owner.

**Work continuously through the end-to-end assignment.** The owner authorizes the agent to select and implement one proportionate optimization within this scope without an intermediate approval. Record the options, evidence, tradeoffs, intended files and tests before editing; choose the simplest candidate supported by evidence. Avoid broad architecture changes. If no candidate has sufficient evidence, improve measurement or report the limitation instead of inventing an optimization. Follow existing AWS authorization and the $5 total target. Do not push, publish, commit, install dependencies, expose services, or perform destructive actions without applicable authorization. Do not treat this document as permission to upload source or logs to an external model.

Persist across turns when the agent runtime supports it. When a check fails or a tool is missing, investigate the cause, try a safe supported alternative, and continue independent tasks. Keep a continuation record in FINDINGS so another session can resume. Do not repeat the same failing action indefinitely or create billable resources when cost or cleanup is uncertain. Pause and report a specific blocker only when safe progress cannot continue, a required permission is missing, or the user's budget or runtime limits are reached. Never fabricate success to end the task.

Protect the user's device and account: do not delete or overwrite user data, delete accounts, destroy unrelated cloud resources, change credentials or permissions, or bypass security controls. The existing authorization to terminate and remove only the temporary AWS resources created by a benchmark remains essential for cleanup; verify their recorded identities and ownership first. Respect the $5 total project target, which is not a guaranteed AWS billing cap. Model/API usage may have separate costs and must follow the user's account limits.

## Starting evidence: verify before relying on it

- Current handoff was prepared at commit `9c7ab13`; inspect `git status --short --branch` and `git log -5 --oneline` first. Preserve uncommitted work. Later changes may supersede these facts.
- [Instrumented AWS run](../instrumented-aws-run.md): one `c6i.large`, one gateway and one CPU worker with the client on the same host; 360 requests passed response-shape checks. At concurrency 1, 2, and 4, unique-input request-weighted median whole-batch backend times were approximately 196, 372, and 660 ms. Queue medians were about 20 ms. These are not isolated model execution times or proof of a cause.
- [Baseline](../aws-baseline.md) and [timing definitions](../timing-instrumentation.md) explain the synthetic padded inputs, small sample sizes, shared CPU, cache distinction, and repeated per-batch timing values. Never sum request-weighted batch timings as independent batch work.
- [Cost ledger](../aws-cost-ledger.md) reports $0.009615 in estimated prior project resource costs, excluding items and billing adjustments; it is not a verified bill or a hard cap. Check the latest ledger and cost information before paid work. The owner set a **$5 total project target**, not a fresh $5 per run; unknown costs reduce the safe available budget.
- The AWS source-delivery mechanism currently permits source changes only in `include/batch_processor.h` and `src/worker_node.cpp`; other `src/` or `include/` changes are rejected. A backend change requires an explicit, tested source-delivery extension or another reviewed reproducible method; never silently benchmark different code. See [run guide](../../benchmark-aws/README.md).
- Current code sets ONNX Runtime intra-op threads to 4, uses a 20 ms batch deadline and maximum batch size 32, serializes backend calls with a mutex, and prepares padded batch inputs. These are investigation leads, not proven defects. Read the actual code before drawing conclusions.

## Deliverables and acceptance criteria

Apply the evidence and completion gates in [ACCEPTANCE.md](ACCEPTANCE.md). Continue through the whole assignment when safe and authorized; report any unmet gate as blocked with a resumable handoff.

1. A completed findings record with scope, source revision, environment, commands, evidence locations, facts, hypotheses, blocked checks, and risks. Exclude secrets and private data.
2. A diagram or concise request path from client through gateway, routing, cache, batching, backend, and response, with the boundaries of existing timers.
3. A ranked hypothesis table with supporting and opposing evidence, proposed discriminating experiments, expected effort, and correctness risk. Include backend runtime versus preparation, thread oversubscription, batching, HTTP/gateway overhead, caching, and measurement artifacts.
4. A reproducible local baseline when dependencies already exist; otherwise report precise blockers and complete all independent static analysis. No fabricated test or benchmark results.
5. Focused findings for correctness, concurrency, failure handling, security, and AWS cleanup that could invalidate measurements or cause harm. Rank by impact and evidence; distinguish suspected issues from reproduced defects.
6. One evidence-supported improvement and at least one alternative, with a documented selection, relevant tests, controlled before/after measurements when possible, and an explicit no-improvement outcome where applicable.
7. A two-day task breakdown, stopping points, and a resume-safe description limited to verified work. Do not claim production readiness or distributed scaling from single-host measurements.
8. A reviewable, narrowly scoped implementation with relevant tests; a reproducible before/after comparison when feasible; and a short public-facing summary stating the owner's contributions, upstream attribution, measurement conditions, results, and limitations. If the change or comparison is blocked, report the gap and retain evidence rather than inventing results.
9. A clean delivery review following [repository hygiene](REPOSITORY-HYGIENE.md): only purposeful files, no secrets or raw sensitive logs, no unrelated changes, and no Git publication or commits without the applicable authorization. Report the exact final status and any existing user changes.

## Workflow and timeboxes

### Phase 0 — orient and protect state (30–45 minutes)

1. Read applicable `AGENTS.md`, project [scope](../benchmark-scope.md), [AWS guide](../../benchmark-aws/README.md), timing and result reports, and this plan. Record repository status, HEAD, dependency versions available locally, and whether the raw ignored run directories still exist. Do not print credentials or environment secrets.
2. Identify local ports and system load before running services. Use loopback and temporary directories. Do not stop unrelated processes or alter unrelated repositories. Review scripts before executing them.
3. Check tool availability with `command -v` for `python3`, `cmake`, `ctest`, a C++ compiler, `aws`, and optional profilers. Inspect model and build artifacts by path and checksum if present. Do not install missing dependencies without authorization.

### Phase 1 — map execution and measurement (60–90 minutes)

1. Trace `benchmark-aws/load.py`, `src/gateway.cpp`, `src/worker_node.cpp`, `include/batch_processor.h`, `src/engine_backend.cpp`, and `src/inference_engine.cpp`. Record request, retry, cache, batching, locking, tensor preparation, ONNX Runtime call, output splitting, and response paths. Explain what each timer includes and excludes.
2. Inspect configuration, pinned dependencies and model, `benchmark-aws/remote.sh`, controller source snapshot rules, build flags, and test discovery. Verify that the documented revision and benchmark source correspond to local retained manifests if available. Check whether the client workload actually produces unique and repeated inputs as described.
3. Investigate the apparent difference between end-to-end and queue/backend measurements using per-request evidence, not subtraction of independently aggregated medians. Include scheduling, gateway and worker HTTP, client contention, serialization, and timer boundaries as possible explanations. Do not infer network latency from a shared host.

### Phase 2 — establish a reproducible baseline (1–2 hours)

1. Run available non-mutating checks from the project root: `python3 -m unittest discover -s tests -v` and `bash -n benchmark-aws/remote.sh`. Record passes, failures, and skips. Only run `ctest --test-dir <existing-build-dir> --output-on-failure` if a suitable build already exists; report absent builds or libraries. Review the commands in the run guide before using optional tooling.
2. If a working build, model, and safe loopback ports are available, perform a bounded smoke test with real output and request correspondence checks. Record exact command, revision, hardware, OS, build mode, model hash, runtime version, worker count, thread settings, cache state, warm-up, timeouts, workload, counts, and errors. Stop only processes created by this investigation. Do not compare local numbers directly with EC2 numbers as an improvement.
3. Check numerical behavior rather than response shape alone: compare identical inputs and output dimensions and values to an appropriate baseline with stated tolerances; inspect batch versus single-request behavior. Record missing model-ground-truth data as a limitation.

### Phase 3 — investigate competing causes (2–4 hours)

1. Build an evidence table for: input copy/padding and allocations; ONNX `Session::Run`; output copying; mutex wait; queue deadline and batch size; ORT thread count versus available CPUs; HTTP connection handling and serialization; cache hit/miss and input identity; client/server contention; warm-up and CPU variation. Mark each as observed, plausible, refuted, or untested.
2. Prefer existing logs and non-invasive profiling. If tools and permissions allow, collect bounded CPU, memory, thread, and call-stack measurements locally. Record profiler overhead and sample counts. If additional instrumentation would be needed, specify its exact boundaries, output, overhead control, and tests as a proposed follow-up; do not treat this plan as permission to change the implementation.
3. Investigate suspicious shapes, oversized input handling, output bounds, shutdown with queued requests, exceptions, thread safety, cache semantics, retries and timeouts. Reproduce safely where practical. Treat code-reading concerns as hypotheses until tested. Do not weaken tests to make a check pass.
4. Rank candidates by evidence strength, expected latency or throughput benefit, risk to numerical correctness, complexity, and ability to verify in the remaining time. Pick at most one for the next implementation. No change is a valid conclusion if evidence is insufficient.

### Phase 4 — select a change (30–60 minutes)

In [FINDINGS.md](FINDINGS.md), compare the leading option with a simpler alternative and doing nothing. Record costs and benefits, risks, dependencies, rollback, acceptance criteria, and the evidence that supports the choice. Then select one bounded change and proceed. Ask the owner only if a required action exceeds the existing authorization, budget, or scope, or if a genuinely material choice cannot be made safely from available evidence. Continue independent work while blocked.

## Phase 5 — implementation, comparison, and analysis

- Day 1: investigate, record the evidence, and select one change. Day 2: implement it, add behavioral and numerical regression tests, self-review, compare, and analyze. Allocate time for the final review and handoff. If time expires, hand off the current evidence rather than claiming completion.
- Before changing the AWS delivery mechanism, test that it snapshots every changed source file, rejects untracked or omitted relevant changes, verifies remote hashes, and preserves the existing cleanup and failure checks. Keep the pinned baseline and changed source identities explicit.
- Use paired baseline and changed runs on equivalent hardware and configuration, same model checksum and runtime, same workload, cache state, warm-up, request counts, concurrency and duration. Repeat where budget and time permit; alternate run order to reduce order effects. Record distributions across runs, individual errors, p50/p95, sample counts and throughput. Treat p99 from small samples as unstable. Report cost and performance regressions as well as wins. Preserve raw records and provenance locally.
- Keep instrumentation consistent or measure its overhead separately. Verify numerical outputs and response association across versions. Before running the comparison, record a justified decision rule based on baseline variability, correctness, cost and practical relevance; do not invent an arbitrary win threshold or stop after a favorable sample. Report inconclusive results if the data cannot distinguish change from noise.
- Analyze results after the experiments: compare absolute and relative latency and throughput with sample counts and run-to-run variation, report errors and correctness, explain potential confounders and whether the hypothesis was supported, and state precisely what the experiment cannot establish. Investigate unexpected regressions within the timebox, preserving both favorable and unfavorable results. Do not claim a percentage improvement without matched measurements and a defined baseline.
- Prepare a concise portfolio evidence table: claim, exact personal contribution, command or artifact that supports it, comparison conditions, result, and caveats. Include upstream authorship and distinguish AI assistance from the owner's role where relevant. Use evidence already verified in the project, not project ambitions, as the basis for any suggested resume bullet.
- Update the existing README or a single linked results report only when there is verified new information. Explain how to run and test the work, describe the architecture and change, link a sanitized summary, state sample size and limitations, and clearly label inherited upstream results. Keep raw artifacts local; do not add promotional claims, duplicate reports, or unverified charts.
- At handoff, explain the problem, hypothesis, chosen approach and alternatives, actual change, exact checks and results, benchmark setup and outcomes, remaining risks, and one or two evidence-backed draft bullets. If the measurements do not support a performance claim, use an accurate contribution-focused bullet or state that a quantified bullet is not yet available.

## Phase 6 — update the software engineering resume

The owner requests this as part of the same end-to-end assignment. After completing the project work and analyzing the evidence, update the maintained generic source `/home/althaaf/JOB_SEARCH/MAIN_DOCUMENTS/GENERIC/resume_software_engineer.tex`. Read the applicable workspace instructions and `/home/althaaf/.agents/skills/resume-production/SKILL.md` first. Inspect the source and preserve existing user changes. This specific resume edit is authorized; do not modify other resume families, applications, or archives.

- Add one Projects entry for this distributed inference project, with concise bullets based on verified completed contributions and measurements. Preserve accurate upstream attribution and do not claim a speedup, authorship of inherited features, or multi-host results without evidence. If the investigation or benchmark is incomplete, write only what is supported and explain the limitation.
- In the Experience section retain the Kolosal.AI entry and remove the other experience entries from this resume only. Do not delete underlying evidence or other resume files. Retain relevant education and skills, and keep other projects unless space and relevance require a documented adjustment.
- Produce and verify the one-page resume PDF and its certificate-appended version, using the existing software-engineer family names and the workspace's required `resume.pdf` and `resume_with_IPhO.pdf` deliverables where applicable. Inspect the generic README and current file conventions before overwriting any shared default PDF; if ownership is unclear, preserve it and report the ambiguity. Compile in a temporary build directory when possible. Check extracted text, links, page count, layout, and the skill's bullet-tail rule; append the IPhO certificate as a separate final page. Report exact artifact paths and checks or blockers.
- Keep resume source, personal information, and PDFs out of the project repository and out of external services unless separately authorized. Do not publish or commit the resume. Include the resume update and verification status in the final handoff.

## AWS and spending gate

The owner previously allowed multiple benchmark instances within a total $5 usage target, with cost calculated after each operation and cleanup. That permission does not override current tool policy or authorize unrelated resources, external data disclosure, or unlimited experimentation. Before any new cloud action, check active authorization and account context, the latest estimated and available billed project costs, current rates, maximum run duration, expected upper cost including storage, IPv4 and transfer, and a contingency reserve. AWS budgets and estimates are not hard spending caps; stop if the remaining budget or cleanup is uncertain.

Use the existing preflight and controlled runner only after inspecting them. Keep ingress restricted, retain logs locally, checksum artifacts, tag only owned resources, terminate the run's resources, independently verify cleanup where possible, and update the cost ledger after each paid operation. If collection or cleanup fails, stop further runs, retain evidence, follow the documented recovery procedure, and notify the owner. Do not make unattended overnight runs unless the owner specifically approves the operational conditions and a reliable operator/cleanup path.

## Explicit non-goals

No public service, production deployment, new paid services, GPU, Kubernetes, redesign into microservices, real customer data, broad refactor, unsupported accuracy claims, or multi-host scaling claims. Multi-host benchmarking is a future decision with separate cost, security, and measurement design. Do not change upstream attribution or present the original author's results as this project's own.
