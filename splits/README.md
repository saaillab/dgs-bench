# DGS-Bench Fixed Splits

This directory stores the exact deterministic split definitions promised in Section 7 of the DGS-Bench camera-ready paper:

- `cassava_10seed_splits.json`: `train.csv` SHA-256 verification hash, class mapping (`0: CBB, 1: CBSD, 2: CGM, 3: CMD, 4: Healthy`), and exact `image_id` lists for `lab_train` (`n = 1,600`, 320/class) and `lab_test` (`n = 400`, 80/class) across all 10 seeds (`42..51`). For each seed, `field_test` (`n = 19,397`) is the exact complement `(all train.csv image_ids) \\ (lab_train U lab_test)`.
- `prevalence_sweep_specs.json`: Exact target dominant-class prevalences (`0.20..0.658`), random seed, observed mirage onset per architecture, and closed-form threshold values $p^*$.

To regenerate split indices programmatically from any compatible label array:

```python
from dgs_bench import create_dgs_splits

train_idx, test_idx, field_idx = create_dgs_splits(labels, seed=42, n_per_class=400)
```
