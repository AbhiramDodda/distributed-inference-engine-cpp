# Temporary EC2 benchmark

Read [scope and acceptance](../docs/benchmark-scope.md) before running. The controller creates billable AWS resources. It uses the default AWS CLI profile unless `AWS_PROFILE` is set, in `us-east-1` only. It expects AWS CLI, Python 3.12+, OpenSSH, and HTTPS access to AWS and GitHub.

The controller uses an ephemeral security group allowing SSH only from the detected local public IPv4 address, a local ephemeral SSH key, an EC2 instance and its root volume. It does not open application ports. It retrieves the SSH host key through the authenticated EC2 console API. It schedules a 60-minute instance-initiated termination and limits the workload to 30 minutes. The local machine and AWS credentials must remain available for log collection and cleanup.

Run from the repository root:

```sh
python3 benchmark-aws/controller.py preflight
python3 benchmark-aws/controller.py run --source instrumented
# For the original upstream comparison:
python3 benchmark-aws/controller.py run --source baseline
```

Review preflight output before `run`. The controller stores logs and results under `benchmark-aws/runs/<run-id>/`, which is ignored by Git. There is no automatic scheduling: start the command when the benchmark should run. If the machine sleeps or loses access, collection may fail. Billing alerts are delayed and do not enforce the $5 target. The public IPv4 address observed at launch must remain valid for SSH during the run.

## Recovery

The run directory contains `state.json` with the run tag and any known instance, volume and security group IDs. If the controller reports cleanup failure or is interrupted, use the same AWS profile and region to inspect **only the recorded run tag and resources**. Run `python3 benchmark-aws/controller.py cleanup --state <run-directory>/state.json`; it verifies run tags before deletion. Check the printed final state and the EC2 console. If the controller is killed before recording IDs, the unique run tag remains in `state.json`, and cleanup discovers tagged resources. Do not terminate unrelated resources.

AWS account policies, network failures, instance boot failure, or process kill can prevent cleanup. The scheduled in-instance shutdown is not guaranteed to run. Manually inspect the account if cleanup is not confirmed.

A successful controller exit requires a transferred archive with a verified checksum, a successful remote workload, and verified cleanup. Missing or partial logs remain in the run directory with a nonzero exit status.

## How the controller is organized

`preflight()` checks the account, image, subnet, and local address. `provision_instance()` records the run ID and creates the security group and instance. `connect_to_instance()` verifies the SSH host key using EC2 console output. `execute_and_collect()` runs the workload and verifies the returned archive. `recover_logs()` attempts to retrieve partial evidence on failure. Finally, `cleanup()` checks the run tag, requests termination, waits for the recorded instance ID to report `terminated`, and verifies the recorded volume and security group are gone. If an AWS response is missing or a deletion cannot be confirmed, it returns failure and leaves the run state for recovery.

For local checks, install the pinned development tool in an isolated environment or use `uvx`:

```sh
uvx --from ruff==0.17.3 ruff check benchmark-aws/controller.py benchmark-aws/load.py benchmark-aws/verify_source.py tests/test_benchmark_setup.py tests/test_source_delivery.py tests/test_instrumented_worker.py
uvx --from ruff==0.17.3 ruff format --check benchmark-aws/controller.py benchmark-aws/load.py benchmark-aws/verify_source.py tests/test_benchmark_setup.py tests/test_source_delivery.py tests/test_instrumented_worker.py
python3 -m unittest discover -s tests -v
bash -n benchmark-aws/remote.sh
```

## Timing instrumentation and AWS source

The benchmark client records worker queue and batch timing fields; see [timing instrumentation](../docs/timing-instrumentation.md). Before creating AWS resources, the runner snapshots the diff for `include/batch_processor.h` and `src/worker_node.cpp`, plus the benchmark scripts. The remote host checks the patch, scripts, and resulting source hashes before building. Instrumented runs fail if worker timing measurements are absent. Baseline mode uses the same pinned upstream revision with an empty patch. Run records retain the exact patch and manifest; review these when comparing results. The pinned baseline must be an ancestor of local HEAD. Changes to other files under `src/` or `include/` are rejected rather than silently omitted; the remote build uses the baseline CMake configuration.
