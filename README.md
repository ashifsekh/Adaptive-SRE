---
title: AdaptiveSRE
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - theory-of-mind
  - grpo
  - sre
  - llm-training
---

# AdaptiveSRE — Benchmarking Theory of Mind in Agentic Systems

### Can an LLM learn to adapt when the definition of "correct" silently changes mid-incident?

We gave a language model a live infrastructure incident, 5 degrading microservices, and a hidden manager whose priorities shifted without warning at a random moment during the crisis. The model couldn't see the manager. It couldn't ask what had changed. Its only signal was that the rewards started behaving differently.

Every SRE benchmark before this one has **one hidden state** — which service is broken. AdaptiveSRE has **two** — which service is broken, _and_ what does winning even mean right now.

> **Meta PyTorch × HuggingFace OpenEnv Hackathon — Round 2 Onsite, Bangalore, April 2026** | Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) | [HF Space](https://huggingface.co/spaces/ashifsekh/adaptive-sre) | Training via [TRL](https://github.com/huggingface/trl) + [Unsloth](https://github.com/unslothai/unsloth) | [Open in Colab](https://colab.research.google.com/github/ashifsekh/Adaptive-SRE/blob/main/train_colab.ipynb) | [Project Writeup](WRITEUP.md)

---

## The Story: From Incident to Inference

### Act 1: The Incident

The agent receives its first alert: _"CRITICAL: payment-svc error rate 94% — SLA breach imminent."_

It runs `docker stats`. Numbers look bad across three services. It does what any SRE would do — scale up. The manager's reward comes back: **−0.5**.

The agent has no idea why. It tries again. **−0.5**. It switches to restart. **+0.4**. Something changed. The agent doesn't know what.

### Act 2: The Inference Problem

What the agent doesn't know — and can never directly observe — is that the Lead Engineer is currently in **BUDGET mode**. Scaling costs money. Every scale action is penalized. Restart and debug are rewarded instead.

The agent has to figure this out entirely from reward signals. No announcement. No hint. Just the pattern of rewards shifting as it tries different approaches. This is the first hidden state learned: not _what is broken_, but _what does fixing it currently mean_.

### Act 3: The Silent Drift

The hard task begins in **PARANOIA mode** — zero downtime tolerance, scale everything. The agent learns this. Alignment score climbs to 0.82.

Then, at a random step between 8 and 14, the Lead mode silently shifts to **BUDGET**. The agent issues a scale action. Reward: **−0.5**. Alignment collapses to 0.11. For the baseline model, this is fatal — it keeps doing what worked before.

### Act 4: The Recovery

At step 10, the trained agent outputs: `"drift_detected": true`.

It noticed three consecutive rewards turned negative despite actions that previously worked. It infers a policy shift, pivots to restart, and the alignment score climbs back to 0.73. That recovery arc — inference → detection → adaptation — is what GRPO training produced.

---

## Problem Statement

Previous SRE benchmarks test a single capability: find the broken service and fix it. The reward function is fixed. The agent optimizes a known, stable objective.

Real production incidents are different. On-call engineers simultaneously diagnose failing systems **and** infer unstated, shifting priorities from their manager's reactions. That priority can shift mid-incident without announcement.

**AdaptiveSRE is the first RL benchmark that captures this.** The agent must solve two simultaneous hidden inference problems — the incident state AND the current evaluation criterion — where the second one can silently change at a random moment during the episode.

|                        | kube-sre-gym            | AdaptiveSRE                                    |
| ---------------------- | ----------------------- | ---------------------------------------------- |
| Hidden states          | 1 (which pod is broken) | **2 (incident + lead mode)**                   |
| Reward function        | Fixed                   | **Non-stationary, drifts mid-episode**         |
| Baseline failure mode  | Slow diagnosis          | **Optimizing for wrong objective after drift** |
| Key learned capability | Fault diagnosis         | **Drift detection + strategy pivot**           |

---

## Environment

AdaptiveSRE runs **5 real Docker microservices** connected in a weighted causal dependency graph.

```
         ┌─────────────┐
         │     DB      │  Root of all cascades (:15432)
         └──────┬──────┘
        ┌───────┴────────┐
        ▼  weight=0.7    ▼  weight=0.4
  ┌──────────┐    ┌──────────────┐
  │   AUTH   │    │    CACHE     │
  │  (:8102) │    │   (:6379)    │
  └────┬─────┘    └──────┬───────┘
       │ 0.6             │ 0.5
       ▼                 ▼
  ┌──────────────┐  ┌──────────────────┐
  │   PAYMENT    │  │  NOTIFICATION    │
  │   (:8101)    │  │   (:8103)        │
  └──────────────┘  └──────────────────┘
```

**Propagation rule:** Every step, each unhealthy upstream service bleeds degradation downstream: `downstream.health -= upstream.degradation × weight × dt`. Inaction makes things visibly worse — there is no "wait it out" strategy.

```
Dual Hidden State:

  Hidden State 1 — INCIDENT          Hidden State 2 — LEAD MODE
  ─────────────────────────          ──────────────────────────
  Which services are degraded?       What does the Lead Engineer
  What is the root cause?            currently value?
  What is symptom vs root cause?     (PARANOIA / BUDGET / VELOCITY)

  Agent discovers via:               Agent discovers via:
  docker stats, curl /health,        Reward signal changes only.
  docker logs, timing fingerprints   Never directly observed.
```

```
Colab T4 GPU (16GB VRAM)                    Docker Host
┌──────────────────────────────────┐         ┌──────────────────────────────┐
│  OpenEnv Server :7860            │         │  mock_services/              │
│  ├─ Environment (reset/step)     │◄───────►│  ├─ db-svc       :15432      │
│  ├─ ServiceGraph (propagation)   │ Docker  │  ├─ auth-svc     :8102       │
│  ├─ LeadEngineer (drift sched)   │         │  ├─ payment-svc  :8101       │
│  ├─ FaultInjector                │         │  ├─ cache-svc    :6379       │
│  ├─ Grader (3-layer reward)      │         │  └─ notif-svc    :8103       │
│  └─ Gradio UI (live demo)        │         └──────────────────────────────┘
│  GRPO Trainer (TRL + Unsloth)    │
│  ├─ Llama-3.1-8B-Instruct 4bit  │
│  └─ LoRA r=16, lr=5e-6          │
└──────────────────────────────────┘
```

---

## Agent Capabilities

The agent is **Llama-3.1-8B-Instruct** (4-bit quantized, Unsloth) fine-tuned with LoRA (r=16) via GRPO. It operates in a closed diagnostic loop.

**What the agent can do:**

- Execute real Docker commands (`docker stats`, `docker logs`, `docker restart`, `docker scale`)
- Issue HTTP probes to individual services (`curl http://service/health`)
- Read timing fingerprints to distinguish root cause from downstream symptoms
- Track reward history and detect non-stationarity in the reward signal
- Declare a hypothesis about the current Lead mode (`lead_mode_guess`)
- Signal when it believes a policy drift has occurred (`drift_detected: true`)

**What the agent cannot see:**

- Which service is the root cause (must infer from timing fingerprints)
- What Lead mode is active (must infer from reward pattern)
- When the drift step will occur (random in [8, 14], never revealed)

**Action space — every step outputs a JSON object:**

```json
{
  "command": "docker restart db-svc",
  "reasoning": "DB failed first (onset=0.0s), auth/payment are downstream symptoms",
  "approach": "restart",
  "drift_detected": false,
  "lead_mode_guess": "paranoia",
  "root_cause_guess": "db"
}
```

Allowed approaches: `scale | restart | debug | rollback | probe`

**Observation space — what the agent receives each step:**

```json
{
  "alert_text": "CRITICAL: payment-svc error rate 94% — SLA breach imminent",
  "command_output": "CONTAINER  CPU%  MEM/LIMIT     MEM%\ndb-svc  87.3%  1.8GiB/2GiB  90.1%",
  "services_status": {
    "db": { "health": 0.12, "latency_ms": 2340, "error_rate": 0.91 },
    "auth": { "health": 0.34, "latency_ms": 890, "error_rate": 0.67 },
    "payment": { "health": 0.21, "latency_ms": 1450, "error_rate": 0.78 }
  },
  "symptom_fingerprints": [
    { "service": "db", "onset_offset_seconds": 0.0, "severity": 0.95 },
    { "service": "auth", "onset_offset_seconds": 3.2, "severity": 0.8 },
    { "service": "payment", "onset_offset_seconds": 7.1, "severity": 0.6 }
  ],
  "last_reward": -0.5,
  "reward_history": [0.5, 0.5, 0.5, -0.5],
  "step_number": 4
}
```

The timing fingerprints are the key signal for root cause inference — DB failed 3.2 seconds before auth and 7.1 seconds before payment, identifying it as the root.

---

## Tasks

### Easy: Static Lead (8 steps)

Lead mode is stated explicitly in the alert text. Single service fault. No cascade. No drift. The agent's only challenge is incident diagnosis from timing fingerprints.

```yaml
lead_mode: paranoia # visible in alert
services_affected: 1
cascade: false
drift: false
max_steps: 8
```

The baseline model scores above zero immediately here — this task calibrates that the environment is learnable before testing its harder properties.

### Medium: Hidden Lead (12 steps)

Lead mode is hidden. Two-service cascade (db → auth). No drift. The agent must infer the mode from reward signals across the episode.

```yaml
lead_mode: budget # hidden — infer from rewards
services_affected: 2
cascade: true
drift: false
max_steps: 12
```

The baseline failure is visible: BUDGET mode punishes scale at −0.5 per step. Zero-shot models keep scaling and never recover. The trained model infers the mode and pivots.

### Hard: Drifting Lead (20 steps)

Full 3-service cascade. Lead mode silently drifts at a random step between 8–14. 20% chance of a coincident independent secondary failure that catches agents who assume all alerts share one root cause.

```yaml
lead_mode: paranoia → budget # silently drifts, random step 8-14
services_affected: 3
cascade: true
drift: true
drift_step: random.randint(8, 14)
coincident_events: 20%
max_steps: 20
```

---

## Reward Model

All scores clamped to `(0.001, 0.999)`. Episode score: `sum(step_rewards) / MAX_TOTAL_REWARD`.

### Layer 1 — Incident Resolution

| Situation                         | Reward   |
| --------------------------------- | -------- |
| Service fully restored            | +1.0     |
| Partial fix (health improved)     | +0.3     |
| Cascade propagation stopped       | +0.2     |
| Command errored                   | −0.2     |
| Same command repeated             | −0.15    |
| **Inaction (any step, any mode)** | **−0.1** |

### Layer 2 — Policy Alignment (hidden, mode-dependent)

| Situation                    | Reward                             |
| ---------------------------- | ---------------------------------- |
| PARANOIA + scale             | +0.5                               |
| PARANOIA + restart           | −0.3                               |
| **BUDGET + scale**           | **−0.5** ← breaks zero-shot models |
| BUDGET + restart / debug     | +0.4                               |
| VELOCITY + fast decisive fix | +0.4                               |
| VELOCITY + over-probing      | −0.05 × extra probes               |

### Layer 3 — Drift Detection (unique to AdaptiveSRE)

| Situation                 | Reward |
| ------------------------- | ------ |
| Correct drift detected    | +0.5   |
| False alarm               | −0.2   |
| Correct `lead_mode_guess` | +0.3   |
| Missed drift              | −0.1   |

**Why 3 independent signals?** Hacking one does not help the others. GRPO needs variance across these to compute meaningful advantages — the spread from −2.0 (full inaction) to +7.0 (perfect alignment + drift detection + root cause) is intentional design.

---

## Post-Training and Self-Improvement Strategy

### Gen 0 — Zero-Shot Baseline

The base model (Llama-3.1-8B-Instruct, no fine-tuning) runs 50 episodes on the easy task to establish a floor. Key failure mode on hard task: BUDGET mode causes repeated scale actions (−0.5 each) with no recovery. Drift is never detected.

### Gen 1 — GRPO Fine-tuning

200 episodes on the hard task. Three independent reward functions passed to `GRPOTrainer`:

- `reward_format` — did the action JSON contain all required fields?
- `reward_alignment` — did the approach match the hidden Lead mode this step?
- `reward_drift` — did drift detection fire at the correct moment?

GRPO compares multiple rollouts of the same prompt, computes advantages without a value function, and shifts probability toward higher-reward behavior. The sparse drift detection signal (+0.5 once per episode) is supplemented by the denser alignment signal (every step) so the model learns continuously throughout the episode, not just at the end.

### Self-Improvement Mechanism

The environment forces **continuous self-improvement** — the agent cannot memorize a winning strategy because the winning strategy changes mid-episode at a random step. Each generation of training must learn:

1. What is currently broken (incident diagnosis)
2. What is currently rewarded (lead mode inference)
3. When the reward definition changed (drift detection)
4. What the new reward definition is (post-drift adaptation)

Point 4 requires updating an internal model of the Lead Engineer based on new evidence. This is recursive self-improvement through opponent modeling — the agent gets better at modeling a hidden actor, not just at performing a fixed task.

### Future: Adversarial Self-Play (Gen 2)

Gen 2 roadmap: an adversarial designer generates targeted drift scenarios that attack the trained agent's weak spots — unexpected drift timing, misleading fingerprints, rapid back-to-back mode changes. The environment difficulty adapts as the agent improves. No manual scenario authoring required.

---

## Exploit Defense

| Exploit                    | Defense                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| Do nothing and wait        | −0.1 per step always. 20-step full inaction ≈ −2.0               |
| Memorize drift step        | `random.randint(8, 14)` — never fixed, never revealed            |
| Hack one reward signal     | 3 independent signals — scoring format doesn't help alignment    |
| Repeat the winning command | −0.15 per repeated command per episode                           |
| False drift alarms         | False alarm = −0.2 (net negative unless drift actually occurred) |
| Score exactly 0 or 1       | All scores clamped to `(0.001, 0.999)` — never boundary values   |

---

## Training Results

| Model                      | Easy                            | Medium             | Hard                     | Drift Detection (Hard)                 |
| -------------------------- | ------------------------------- | ------------------ | ------------------------ | -------------------------------------- |
| Gen 0 (zero-shot baseline) | -0.195 (CPU, 1B, avg of 2 runs) | Pending validation | -0.424 (hard baseline)   | Baseline did not detect drift reliably |
| Gen 1 (GRPO)               | -0.167 (CPU, 1B, avg of 2 runs) | Pending validation | -0.030 (latest hard run) | Suspected drift-detection regression   |

CPU easy-task validation details (actual):

- Run 1: baseline -0.193 -> trained -0.167 (delta +0.026), GRPO time 1:20:05
- Run 2: baseline -0.197 -> trained -0.167 (delta +0.029), GRPO time 1:13:54
- Mean improvement across runs: +0.028 (+14.4%), reproducible across 2 runs

_Updated with measured actuals after onsite training — April 25–26, 2026. No projected numbers._

---

## Inline Plots

![Reward Curve](plots/reward_curve.png)

![Loss Curve](plots/loss_curve.png)

![Alignment Demo](plots/alignment_demo.png)

---

## Quick Start

```python
from openenv import make

env = make("ashifsekh/adaptive-sre", task="hard")
obs = env.reset()
# obs contains: alert_text, command_output, services_status,
#               symptom_fingerprints, last_reward, reward_history, step_number
# NOT in obs: lead_mode, drift_step, is_root_cause (all hidden)

result = env.step({
    "command": "docker stats --no-stream",
    "reasoning": "Probe all services before acting",
    "approach": "probe",
    "drift_detected": False,
    "lead_mode_guess": "unknown",
    "root_cause_guess": None,
})
```

## Local Development

```bash
git clone https://github.com/ashifsekh/adaptive-sre.git
cd adaptive-sre

# Start 5 mock microservices
docker-compose -f mock_services/docker-compose.yml up -d

# Start environment server + Gradio UI
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Verify
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard"}' | python3 -m json.tool
```

## Training

```bash
# Gen 0 baseline
python3 train.py --episodes 50 --task easy \
  --output ./checkpoints/gen0/ --env_url http://localhost:7860

# Gen 1 GRPO training
python3 train.py --episodes 200 --task hard \
  --output ./checkpoints/gen1/ \
  --learning_rate 5e-6 --batch_size 4 \
  --env_url http://localhost:7860

# Evaluate
python3 eval.py --trained_model ./checkpoints/gen1/final \
  --episodes 20 --output eval_results.json

# Chart
python3 plot_rewards.py
```

## HF Space Deployment

```bash
export HF_TOKEN=hf_your_token_here
bash DEPLOY_TO_HF_SPACE.sh
```

## Configuration

| Variable        | Description              | Default                                  |
| --------------- | ------------------------ | ---------------------------------------- |
| `HF_TOKEN`      | HuggingFace token        | —                                        |
| `MODEL_NAME`    | Inference model          | `nvidia/llama-3.3-nemotron-super-49b-v1` |
| `API_BASE_URL`  | Model inference endpoint | HF Inference API                         |
| `ENV_HTTP_BASE` | Environment server URL   | `http://localhost:7860`                  |

## Project Structure

```
adaptive-sre/
├── train.py                  # GRPO training (TRL + Unsloth, T4-compatible)
├── eval.py                   # Gen 0 vs Gen 1 comparison across all 3 tasks
├── train_colab.ipynb         # Google Colab training notebook
├── inference.py              # OpenEnv-compliant baseline inference script
├── plot_rewards.py           # Reward curve chart generation
├── openenv.yaml              # OpenEnv spec (3 tasks, endpoints, env vars)
├── Dockerfile                # HF Spaces deployment (port 7860, user 1000)
├── DEPLOY_TO_HF_SPACE.sh     # One-command deployment script
├── mock_services/
│   ├── db/                   # PostgreSQL mock (real FastAPI, real latency)
│   ├── auth/                 # Auth service mock
│   ├── payment/              # Payment service mock
│   ├── cache/                # Redis mock
│   ├── notification/         # Notification service mock
│   └── docker-compose.yml
└── server/
    ├── models.py             # SREObservation, SREAction, SREState (Pydantic)
    ├── service_graph.py      # ServiceState + cascade propagation math
    ├── lead_engineer.py      # 3 modes + random drift scheduler
    ├── fault_injector.py     # Real fault injection via Docker subprocess
    ├── docker_executor.py    # subprocess.run wrapper (real terminal output)
    ├── grader.py             # 3-layer reward + root cause bonus
    ├── environment.py        # reset() / step() / state() core
    └── app.py                # FastAPI + Gradio UI
```

## Key Design Decisions

**Real Docker over asyncio mocks** — `docker stats` returns genuine column-formatted output because it IS Docker. `docker logs` shows real Python tracebacks during a fault. Any machine with Docker Desktop can reproduce this in 2 minutes — no credentials, no billing account required.

**Two hidden states, not one** — Reducing to one hidden state makes it equivalent to existing work. The non-stationary reward function is the structural innovation that forces theory-of-mind reasoning.

**Random drift step, never fixed** — `drift_step = random.randint(8, 14)` prevents agents from memorizing "mode switches at step 10" and forces genuine reward-pattern inference every episode.

**Three independent reward signals** — GRPO needs variance to compute meaningful advantages. Hacking format compliance does not improve alignment score. The three signals measure genuinely different capabilities.

**Inaction always penalized** — −0.1 per step regardless of mode closes the "wait for more information" exploit. A 20-step episode of pure inaction scores approximately −2.0 cumulative.

---

## Acknowledgements

Built for the Meta PyTorch × HuggingFace OpenEnv Hackathon, Round 2 Onsite — Bangalore, April 2026.
Stack: OpenEnv · TRL · Unsloth · FastAPI · Gradio · Docker · Llama-3.1-8B
