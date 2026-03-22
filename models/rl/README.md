# RL (Variant A) Offline Training Pipeline

This folder contains a minimal end-to-end pipeline for **Variant A**:

- Generate offline rollouts using the **digital twin** (`DC_digital_twin`)
- Train a conservative offline policy (BC warm-start + conservative critic regularization)
- Evaluate the policy on the twin
- Load the policy into `orchestrator_service` and enable `controller=rl`
- Runtime safety-shield rejects actions when `predict_server_temp` predicts high overheat risk

## 1) Generate offline dataset

Choose a twin config from `DC_digital_twin/config/*.yaml`, then run:

```bash
python3 models/rl/generate_dataset.py \
  --config-path "DC_digital_twin/config/config_google_300s.yaml" \
  --episodes 20 \
  --steps-per-episode 500 \
  --seed 42 \
  --out-path "models/rl/datasets/dataset_google_300s_ep20.npz"
```

This produces:
- `dataset_....npz`
- `dataset_....meta.json` (includes `threshold_c`, `obs_keys`, constraints)

## 2) Train offline policy

Train the conservative actor-critic policy:

```bash
python3 models/rl/train_offline.py \
  --dataset-path "models/rl/datasets/dataset_google_300s_ep20.npz" \
  --out-path "models/rl/artifacts/policy_google_300s_conservative.pt" \
  --seed 42 \
  --device cpu \
  --batch-size 256 \
  --bc-pretrain-updates 2000 \
  --updates 8000
```

The output artifact is a Torch `dict` containing:
- `actor_state_dict`
- observation normalization (`obs_mean`, `obs_std`)
- action bounds (`setpoint_min`, `setpoint_max`)
- `obs_keys`

## 3) Evaluate policy (smoke/benchmark)

```bash
python3 models/rl/evaluate_policy.py \
  --policy-path "models/rl/artifacts/policy_google_300s_conservative.pt" \
  --config-path "DC_digital_twin/config/config_google_300s.yaml" \
  --out-json "models/rl/artifacts/eval_google_300s.json" \
  --episodes 10 \
  --steps-per-episode 500 \
  --seed 123
```

It prints and saves metrics:
- `never_overheat_episode_fraction`
- `never_overheat_step_fraction`
- `max_avg_chip_temp`
- `total_energy_kwh`, `total_cost_eur`, `total_co2_kg`

## 4) Load policy into `orchestrator_service`

Run the services (digital twin + orchestrator + `predict_server_temp` at least).

In `orchestrator_service`, load the policy via API:

1. Select RL controller:
```bash
curl -X POST "http://127.0.0.1:8020/api/v1/controller/select" \
  -H "Content-Type: application/json" \
  -H "X-Role: operator" \
  -d '{"controller":"rl"}'
```

2. Configure/load the policy artifact:
```bash
curl -X PATCH "http://127.0.0.1:8020/api/v1/controller/rl/config" \
  -H "Content-Type: application/json" \
  -H "X-Role: operator" \
  -d '{"model_path":"models/rl/artifacts/policy_google_300s_conservative.pt","version":"v1","deterministic":true}'
```

3. (Optional) Reload policy version without changing the artifact path:
```bash
curl -X POST "http://127.0.0.1:8020/api/v1/controller/rl/reload-policy" \
  -H "X-Role: operator"
```

## 5) Safety-shield behavior

When `controller=rl`, before applying an action the orchestrator does:
- Build `predict_server_temp` input window from the running history in `RuntimeManager._temp_history`
- Override the newest step’s `setpoint` (and `server_fan_speed` approx) with the candidate action
- Compute aggregated `p_overheat` risk
- If `risk > safety.safe_max_overheat_risk`:
  - action is replaced with fallback: `setpoint_min` + `fallback_fan_speed_pct`

Defaults are in `services/orchestrator_service/app/config.py`:
- `safety.safe_max_overheat_risk = 0.2`
- `safety.fallback_fan_speed_pct = 100.0`

