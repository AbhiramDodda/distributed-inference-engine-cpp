# One-session AWS inference benchmark

## Goal and scope

Run an attributable baseline of the upstream CPU ONNX gateway and one worker on a temporary EC2 instance. Save measurements and operational evidence locally. This is a single-host experiment; it does not measure distributed scaling or maximum server capacity.

## Requirements and acceptance

1. Use the existing AWS account, `us-east-1`, a single on-demand `c6i.large`, and an EBS root volume configured for deletion on termination. The $5 USD target covers this baseline and subsequent comparisons; it is not an enforceable AWS billing cap.
2. Run repeated-input and distinct-input requests at modest concurrency. Record every request's outcome, latency, cache flag and HTTP status; report successful throughput and successful-request p50/p95/p99 separately from errors. Inputs contain three synthetic floats; the engine pads them to model dimensions.
3. Pin the upstream revision and dependency versions. Record model checksum, host details, exact configuration, output, service logs, timing and exit status. Do not record credentials or inference payloads.
4. Restrict network ingress to SSH from the local public IP. Do not expose the gateway or worker ports. Verify the ephemeral SSH host key through EC2 console output.
5. Stream the remote execution log locally, retrieve the results archive, verify its checksum and contents, and retain partial logs on error. No design can guarantee every log survives host or network failure.
6. Terminate only resources created for this run, wait for termination, check root volume deletion and security-group removal. An instance-side timed shutdown is a fallback; report any unverified cleanup or transfer explicitly.
7. Run meaningful local tests of the controller's failure and cleanup paths before a cloud run. A real AWS test remains separate evidence.

## Time and follow-up

Tonight: prepare and test the run procedure. Overnight: one bounded run, not an all-night load test. Day 1: analyze baseline, select one justified optimization. Day 2: implement it, run relevant regression checks and compare under the same conditions. The baseline cannot establish an optimization win.

## Risks and choices

- The load client shares the machine with the engine. Capture host CPU and memory; label throughput as system throughput, not the server's maximum.
- Closed-loop load and small sample counts do not characterize open-loop arrivals or precise extreme percentiles. Report sample counts.
- The upstream repository includes gitlinks without `.gitmodules`; its setup script skips existing empty directories. Fetch the explicitly pinned upstream gitlink revisions instead.
- The supplied model receives synthetic padded inputs, not real image preprocessing. Output checks verify response shape and request correspondence, not model accuracy.
- An interrupted local controller, failed cloud-init, denied AWS API call, or network loss can defeat automated cleanup or log recovery. The operator must follow the recovery runbook and inspect AWS. An instance-side shutdown timer is defense in depth, not a spending guarantee.
- C6i avoids T3 CPU credit throttling as a confounder. On 2026-11-02 the AWS Pricing API quoted $0.085/hour for Linux on-demand `c6i.large` and $0.08/GB-month for gp3 storage in N. Virginia. One hour with 12 GB is about $0.0863 before public IPv4, transfer and tax. Prices can change, and failed cleanup can increase cost beyond $5.

## Non-goals

Public deployment, production readiness, GPU, multi-host scaling, production data, Kubernetes, and an optimization before a baseline.
