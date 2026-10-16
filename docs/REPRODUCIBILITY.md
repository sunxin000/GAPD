# Configuration provenance and unresolved differences

The implementation uses the main-run wrappers for public coefficient and gating defaults. Distillation options use the `GAPD_` prefix.

## Core public defaults

| Setting | Default |
|---|---|
| Distillation coefficient | `GAPD_JSD_COEF=0.5` |
| Distillation estimator | `GAPD_KL_MODE=k1`, `GAPD_USE_PG=true` |
| Raw guide clipping | `GAPD_JSD_TOKEN_CLIP=2.0` |
| Teacher turn limit | `GAPD_TEACHER_MAX_TURNS=-1` (all matched turns) |
| State Jaccard threshold | 0.95 in `state_matcher.py` |
| Same-next-state gate | `GAPD_SKIP_SAME_NEXT_STATE=true` |
| Expression-ID-risk gate | `GAPD_SKIP_EXPRESSION_ID_RISK=true` |
| High MID-F1 gate | `GAPD_SKIP_HIGH_MID_F1_THRESHOLD=0.9` |
| Actor learning rate | `actor_lr=1e-6` |
| Batch / mini-batch / responses | 256 / 128 / 5 |
| PPO clip bounds | 0.2 / 0.28 |

## Source defaults versus local paper appendix

The table compares the launcher defaults with the paper appendix.

| Setting | Current source launchers | Paper appendix | Release action |
|---|---|---|---|
| Distilled spans | `GAPD_SUPERVISE_ACTION_ONLY=false` | generated action-token spans | Confirm which setting produced final results |
| GrailQA epochs | 3 | 1 | Record final experiment command |
| WebQSP / GraphQ epochs | 12 | 8 | Record final experiment command |
| KL regularization | reward KL disabled; actor KL loss disabled | coefficient 0.001 | Confirm which KL route and configuration were used |
| Training top-p | 1.0 | 0.99 | Confirm final rollout configuration |
| ODBC pool size | 12 | 48 | Record backend/resource setting |
| SFT learning rate | 1e-5 | 5e-6 | Confirm final SFT command |

`TOTAL_EPOCHS`, `GAPD_SUPERVISE_ACTION_ONLY`, `ROLLOUT_TOP_P`, `SFT_LR`, the existing lower-case KL variables and ODBC pool variables can be overridden. Changing them produces a new explicit configuration; it is not proof that an archived experiment used those values.

The common source launcher defaults to a coefficient of 0.1 when invoked directly. The public `scripts/train.sh` explicitly sets 0.5 to match the three original main-run wrappers. Use the public wrapper for the documented route.

Validation uses `temperature=1.0` and `top_p=0.7` by default; these can be overridden through Hydra arguments.

## Experiment records

Record resolved Hydra configurations, SFT and RL checkpoint hashes/IDs, dataset hashes, package/container versions, GPU type and seeds alongside experiment results.
