# AWS benchmark cost ledger

Budget target: USD $5 for this project. Figures below are estimates for the resources created by these benchmark attempts, not a verified AWS bill or an enforceable spending limit. Costs for other account activity are excluded.

Rates checked on 2026-11-02: AWS Pricing API returned $0.085 per hour for Linux on-demand `c6i.large` and $0.08 per GB-month for gp3 in `us-east-1`. AWS lists $0.005 per hour for an in-use public IPv4 address. Each attempt used one instance, one 12 GB gp3 root volume and one public IPv4 address. See [AWS public IPv4 pricing](https://aws.amazon.com/vpc/pricing/).

| Attempt | Instance runtime used for estimate | Compute | IPv4 | 12 GB gp3 | Estimated subtotal |
| --- | ---: | ---: | ---: | ---: | ---: |
| First: package conflict | 69 seconds | $0.001629 | $0.000096 | $0.000026 | $0.001751 |
| Second: successful baseline | 155 seconds | $0.003660 | $0.000215 | $0.000057 | $0.003932 |
| Third: instrumented timing run | 155 seconds | $0.003660 | $0.000215 | $0.000057 | $0.003932 |
| Fourth: matched baseline | 156 seconds | $0.003683 | $0.000217 | $0.000058 | $0.003958 |
| Fifth: changed source, 2 ORT threads | 148 seconds | $0.003494 | $0.000206 | $0.000055 | $0.003755 |
| Sixth: changed source repeat | 138 seconds | $0.003258 | $0.000192 | $0.000051 | $0.003501 |
| Seventh: matched baseline repeat | 156 seconds | $0.003683 | $0.000217 | $0.000058 | $0.003958 |
| Total | 977 seconds | $0.023067 | $0.001358 | $0.000362 | **$0.024787** |

Method: runtime is the interval from EC2 `LaunchTime` to the recorded user-initiated termination time. Compute = seconds / 3600 × $0.085; IPv4 = seconds / 3600 × $0.005; storage = 12 × $0.08 × seconds / (30 × 24 × 3600). Each instance ran more than the minimum one-minute billing period. Storage and IP lifetimes are approximated by instance runtime; actual provision and deletion can differ. Data transfer, tax, credits, discounts and billing adjustments are not included. AWS billing may arrive later; consult the bill for actual charges.

The controller verified termination after each run. Direct AWS checks confirmed that the recorded instance was terminated and its recorded volume and security group did not exist after each attempt. The first run was `infer-85e0e6c87756c37a`; the second was `infer-b3ee717ce6dc11fe`; the third was `infer-8bb17685b18e63a4`; the fourth was `infer-d4522a282d0dcc42`; the fifth was `infer-1f39a2ac8519f743`; the sixth was `infer-0601374edef6f33d`; the seventh was `infer-ca8db4050033ae08`. No additional AWS resources were launched for local tests. Cost Explorer reported $0.000000 for 2026-11-01 through 2026-11-03 at the time of writing; billing data can be delayed, so the ledger estimates above remain the conservative record.
