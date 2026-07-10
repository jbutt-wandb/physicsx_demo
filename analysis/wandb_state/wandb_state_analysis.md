# W&B Workshop State Analysis

- Generated: `2026-07-10T22:07:38.909733+00:00`
- Project: `wandb_emea/jb_physics_demo`
- Workspace: https://wandb.ai/wandb_emea/jb_physics_demo?nw=lw1a2ov6xhg

## Executive Readout

- Latest run: `r18m52mt`
- Best run by `model/best_selection_score`: `msqgmxvs`
- Quality signal: model beats baseline on test MSE
- Workspace has profile/slice panels: `True`

## Latest Run

- URL: https://wandb.ai/wandb_emea/jb_physics_demo/runs/r18m52mt
- Created: `2026-07-10T22:03:28Z`
- Files: `297` total, `223` images, `9` table keys, `6` videos
- Media summary keys: `151`

| Metric | Value |
|---|---:|
| `model/best_selection_score` | `0.03281482681632042` |
| `baseline/best_selection_score` | `0.06061510927975178` |
| `model/baseline_guardrail_passed` | `True` |
| `model/best_epoch` | `37` |
| `test/mse` | `0.04988926649093628` |
| `test/cl_rmse` | `0.05256107822060585` |
| `test/cd_rmse` | `0.012131225317716599` |
| `test/cm_rmse` | `0.06779228895902634` |
| `baseline/test_mse` | `0.05294920690357685` |
| `baseline/test_cl_rmse` | `0.06843402236700058` |
| `baseline/test_cd_rmse` | `0.015923434868454933` |
| `baseline/test_cm_rmse` | `0.062112580984830856` |

## Best Run

- URL: https://wandb.ai/wandb_emea/jb_physics_demo/runs/msqgmxvs
- `model/best_selection_score`: `0.023497890681028366`

## Media Progression

| Run | Created | Images | Tables | Videos | Media Keys |
|---|---|---:|---:|---:|---:|
| `zungpcv0` | `2026-07-10T12:18:42Z` | 18 | 0 | 0 | 26 |
| `z83s2sde` | `2026-07-10T12:45:55Z` | 93 | 0 | 0 | 61 |
| `8c0x27d2` | `2026-07-10T12:49:14Z` | 93 | 0 | 0 | 61 |
| `kh9tlki0` | `2026-07-10T12:57:47Z` | 148 | 2 | 2 | 96 |
| `cevvkko5` | `2026-07-10T13:01:58Z` | 148 | 2 | 2 | 96 |
| `msqgmxvs` | `2026-07-10T13:19:32Z` | 148 | 2 | 2 | 96 |
| `n71krz4c` | `2026-07-10T13:30:48Z` | 220 | 2 | 2 | 137 |
| `wme12who` | `2026-07-10T13:47:18Z` | 220 | 2 | 2 | 137 |
| `lqkq560s` | `2026-07-10T21:17:44Z` | 223 | 6 | 2 | 144 |
| `ijgpaa9m` | `2026-07-10T21:41:17Z` | 223 | 6 | 2 | 144 |
| `m5hpwpq5` | `2026-07-10T22:00:15Z` | 223 | 9 | 6 | 151 |
| `r18m52mt` | `2026-07-10T22:03:28Z` | 223 | 9 | 6 | 151 |

## Registry

- `dataset`: `datasets:production` `v0`, aliases `['latest', 'production', 'real-kpi']`, logged by `lhsr2e82`
- `checkpoint`: `checkpoints:production` `v4`, aliases `['epoch-37', 'latest', 'production']`, logged by `wme12who`
- `tuned`: `model_hyperparameter_tuned:production` `v0`, aliases `['latest', 'production', 'run-f7ch7m60', 'sweep-5s1tdqku', 'sweep-winner']`, logged by `f7ch7m60`
- `baseline`: missing, `CommError: artifact membership 'baselines:production' not found in 'wandb_emea_GCMUP7YCNHOUY/wandb-registry-high_lift_surrogates'`

## Issues

- checkpoint production Registry alias resolves to a run other than the latest visual-rich run
- checkpoint production Registry alias is not the historical best validation checkpoint
- baseline Registry alias is missing

## Workspace

- Name: `PhysicsX Workshop - Real HiLift KPI`
- Section `Dataset Visuals`: `8` panels
- Section `Dataset Tables`: `4` panels
- Section `Model vs Baseline Comparisons`: `4` panels
- Section `Prediction Evolution Videos`: `3` panels
- Section `KPI RMSE`: `3` panels
- Section `Model Selection`: `2` panels
- Section `Profiles and Slice Attribution`: `4` panels
- Section `Prediction Media`: `5` panels

## Recommendation

Use the latest visual-rich run for the live workshop. Relink the checkpoint Registry production alias from that run so Registry lineage, workspace panels, and reports all tell the same story. Keep the profile/slice visuals, but describe them as design-variable-derived visual proxies rather than exact CAD.
