# Setup verification — 2026-11-02

## Evidence

- The checked-out upstream revision is `c4d8a08b72df810ffad0dce6a993cecae22494bb`.
- In a disposable local directory, fetched the two dependency revisions recorded by the upstream Git links and ONNX Runtime 1.23.2. Verified the ONNX archive SHA-256, configured a Release build, and built both C++ executables with CMake. No system packages were installed.
- Ran a local gateway and one worker against the included model. Since port 8000 was already in use, changed only the gateway listen port in the disposable build source. Four repeated-input and four unique-input requests at concurrency two all returned valid responses. This is a smoke test, not a performance result.
- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v` passed 12 tests, including local HTTP behavior, failed requests, account isolation, tag-limited cleanup, failed launch, missing launch outcome, archive checksum and traversal.
- `bash -n benchmark-aws/remote.sh`, Python syntax parsing, and `git diff --check` passed.
- `python3 benchmark-aws/controller.py preflight` found the default account, an available AMI and default subnet, and the local public IPv4 address.
- EC2 RunInstances and CreateSecurityGroup dry runs using the planned instance shape and security tags returned `DryRunOperation`. The AWS IAM simulator, queried read-only with the existing admin profile, reported that the deployment user is allowed the principal EC2 actions for launch, cleanup, and console output on the queried resources. Simulation and dry runs do not prove the complete live operation will succeed.
- The AWS Pricing API quoted $0.085/hour for Linux `c6i.large` and $0.08/GB-month for gp3 in `us-east-1`. The budget remains a target, not a hard billing cap.

## Not verified

At the time of the initial checks, no EC2 resources had been created. See the supervised validation section below for the later attempt.

## Supervised AWS validation — 2026-11-02

- One temporary `c6i.large` was launched at 08:58:27 UTC. Its boot-time watchdog and host-key verification worked, and its remote log and failure archive were transferred to the local `benchmark-aws/runs/infer-85e0e6c87756c37a/` directory.
- The workload failed during `dnf install`: the AMI already had `curl-minimal`, which conflicted with the requested `curl` package. No inference benchmark ran on AWS.
- Termination was requested at 08:59:36 UTC. The controller observed the instance in `terminated`; independent CLI checks by recorded IDs returned `InvalidVolume.NotFound` and `InvalidGroup.NotFound` for its root volume and security group.
- At the quoted $0.085/hour, approximately 69 seconds of compute is approximately $0.0016 before storage, public IP, transfer, taxes, and billing adjustments. This is an estimate, not a verified bill.
- The package command now uses the image's existing `curl-minimal`. A local Amazon Linux 2023 container installed the corrected package list successfully. The complete remote script then built the engine and completed 360/360 requests in the local container. That is not an AWS benchmark result. The container preinstalled the basic AMI utilities `sudo`, `util-linux`, `procps-ng`, `tar`, and `gzip` to match the observed EC2 environment.
- At the end of the first attempt, the corrected package setup and workload had not been rerun on EC2. The later second attempt is recorded below.

## Review and quality checks

The controller now separates resource provisioning, connection, workload execution, result collection, log recovery, and cleanup. It explicitly waits for the recorded instance ID to reach `terminated` and checks ownership before acting. The local tests cover missing or delayed responses, wrong tags, failed launch, archive validation, and failure reporting. Ruff 0.17.3 is pinned for lint and formatting; local checks and 19 tests passed. This was a self-review; no independent review or production readiness claim is implied.

## Second AWS attempt — 2026-11-02

After the user authorized additional attempts within the $5 target, the corrected script ran once on a fresh EC2 instance. The gateway and ONNX CPU worker started; all 360 benchmark requests succeeded. The result archive passed its SHA-256 check and was extracted locally. The controller observed termination and cleanup; separate AWS queries confirmed the recorded instance was terminated and the recorded volume and security group were absent. See [the baseline](aws-baseline.md) and [the cost ledger](aws-cost-ledger.md). These checks do not prove production readiness.

## Source delivery validation — 2026-11-02

The runner now supports `instrumented` and `baseline` source modes. Local tests generated both snapshots from the pinned revision, applied the instrumented patch to a clean temporary checkout, compared source hashes, rejected tampered payloads and unexpected source files, and checked that a failed snapshot does not create AWS resources. Instrumented benchmark mode rejects missing worker timings.

A disposable local Amazon Linux 2023 container verified and applied an instrumented snapshot, built the engine, and completed 360/360 benchmark requests. All unique-input requests recorded queue and batch timing samples. The returned archive, source patch, and source manifest were checked locally. The local verification files are in `benchmark-aws/runs/local-delivery-fevdu5wt/`. This was a local container, not an AWS deployment or performance comparison. No additional AWS resource was created for source-delivery validation.

## Instrumented AWS attempt — 2026-11-02

The instrumented patch was applied to the pinned upstream revision on a temporary EC2 instance, and the resulting source and benchmark script checksums were verified before build. The workload completed 360/360 response-shape checks, with complete timing fields for all 181 cache misses. The result archive matched its recorded SHA-256. The controller and separate AWS queries verified instance termination and the absence of the recorded volume and security group. See [instrumented results](instrumented-aws-run.md) and [cost ledger](aws-cost-ledger.md).
