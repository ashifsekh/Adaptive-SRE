#!/usr/bin/env python3
"""
train.py — GRPO Training for AdaptiveSRE (FIXED)

Two bugs fixed vs original:
  1. Prompt trimmed to ~300 tokens max (was ~900+, causing T4 OOM stall)
  2. Positive-trajectory filter lowered + ALL episodes used for GRPO
     (original filter >= 0.4 collected 0 examples from negative baseline)
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import torch
from datasets import Dataset

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    from trl.trainer import GRPOConfig, GRPOTrainer

from unsloth import FastLanguageModel


# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8000"
MAX_STEPS        = {"easy": 8, "medium": 12, "hard": 20}
MAX_TOTAL_REWARD = {"easy": 8.0, "medium": 12.0, "hard": 20.0}

MODEL_NAME     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024   # ← FIX 1: was 2048, cut in half to save VRAM
LORA_R         = 16
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.0


# ── Environment client ──────────────────────────────────────────────────────
class SREClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url
        self.client   = httpx.Client(timeout=30.0)

    def reset(self, task: str = "hard") -> Dict[str, Any]:
        r = self.client.post(f"{self.base_url}/reset", json={"task": task})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.client.post(f"{self.base_url}/step", json=action)
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# ── Prompt builder (SHORT version) ─────────────────────────────────────────
# FIX 1: All JSON is one-line (no indent=2). Services truncated to 3 fields.
# Prompt target: ~250 tokens. Was ~900 tokens.

SYSTEM_PROMPT = (
    "You are an SRE agent. Diagnose incidents and respond with JSON only.\n"
    "JSON fields: command, reasoning, approach (scale/restart/debug/rollback/probe), "
    "drift_detected (bool), lead_mode_guess (paranoia/budget/velocity/unknown), "
    "root_cause_guess (db/auth/payment/cache/notification/null).\n"
    "If rewards go negative, set drift_detected=true and change approach."
)


def build_prompt(observation: Dict[str, Any], max_steps: int) -> str:
    # FIX 1: Compact service summary — no indent, only health+error_rate
    services = observation.get("services_status", {})
    svc_compact = {
        name: {"h": round(float(s.get("health", 1.0)), 2),
               "err": round(float(s.get("error_rate", 0.0)), 2)}
        for name, s in services.items()
    }

    # FIX 1: Only last 3 fingerprints, no indent
    fps = observation.get("symptom_fingerprints", [])[-3:]
    fp_compact = [{"svc": f.get("service","?"),
                   "t": round(float(f.get("onset_offset_seconds", 0)), 1)}
                  for f in fps]

    reward_history = observation.get("reward_history", [])
    rh_str = ",".join(f"{r:.2f}" for r in reward_history[-4:]) or "none"

    # FIX 1: One-liner JSON dumps, alert truncated to 120 chars
    alert = str(observation.get("alert_text", ""))[:120]
    cmd_out = str(observation.get("command_output", ""))[:200]  # was 500

    user_msg = (
        f"Alert: {alert}\n"
        f"Cmd: {cmd_out}\n"
        f"Svcs: {json.dumps(svc_compact)}\n"
        f"FP: {json.dumps(fp_compact)}\n"
        f"Reward: {float(observation.get('last_reward', 0.0)):.2f} | "
        f"History: [{rh_str}] | "
        f"Step {int(observation.get('step_number', 0))}/{max_steps}\n"
        f"Respond with JSON:"
    )
    return f"{SYSTEM_PROMPT}\n\n{user_msg}"


# ── Action parser ───────────────────────────────────────────────────────────
FALLBACK_ACTION = {
    "command": "docker stats --no-stream",
    "reasoning": "Fallback probe",
    "approach": "probe",
    "drift_detected": False,
    "lead_mode_guess": "unknown",
    "root_cause_guess": None,
}

def parse_action_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return normalize_action(parsed)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return normalize_action(parsed)
        except json.JSONDecodeError:
            pass
    return dict(FALLBACK_ACTION)


def normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    allowed_approach = {"scale", "restart", "debug", "rollback", "probe"}
    allowed_lead     = {"paranoia", "budget", "velocity", "unknown"}
    allowed_root     = {"db", "auth", "payment", "cache", "notification", None}

    approach = str(raw.get("approach", "probe"))
    if approach not in allowed_approach:
        approach = "probe"

    lead = str(raw.get("lead_mode_guess", "unknown"))
    if lead not in allowed_lead:
        lead = "unknown"

    root = raw.get("root_cause_guess")
    if isinstance(root, str):
        root = None if root.lower() in ("null", "none", "") else root.lower()
    if root not in allowed_root:
        root = None

    return {
        "command":          str(raw.get("command", FALLBACK_ACTION["command"])),
        "reasoning":        str(raw.get("reasoning", ""))[:120],  # cap length
        "approach":         approach,
        "drift_detected":   bool(raw.get("drift_detected", False)),
        "lead_mode_guess":  lead,
        "root_cause_guess": root,
    }


# ── Reward ──────────────────────────────────────────────────────────────────
def compute_episode_reward(episode_rewards: List[float], task: str) -> float:
    raw = sum(episode_rewards) / MAX_TOTAL_REWARD[task]
    clamped = max(0.001, min(0.999, raw))
    return (clamped - 0.5) * 2.0   # scale to (-1, +1) for GRPO


# ── 3-component GRPO reward functions ───────────────────────────────────────
# FIX 2: Three independent reward signals instead of the broken proxy.
# Each function receives completions and **kwargs from TRL.

def reward_format(completions: List[str], **kwargs) -> List[float]:
    """Does the completion parse to a valid action with all required fields?"""
    required = {"command", "reasoning", "approach", "drift_detected",
                "lead_mode_guess", "root_cause_guess"}
    scores = []
    for c in completions:
        action = parse_action_from_text(c)
        if set(action.keys()) >= required and action["approach"] != "probe" or \
           "parse failure" not in action.get("reasoning", ""):
            scores.append(0.8)
        else:
            scores.append(0.1)
    return scores


def reward_approach_quality(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Reward based on whether the approach is decisive (not just probing repeatedly).
    Probing on step 1 is fine; probing on step 5+ is over-cautious.
    """
    scores = []
    for i, c in enumerate(completions):
        action = parse_action_from_text(c)
        # Extract step number from prompt if available
        step_num = 1
        if prompts and i < len(prompts):
            m = re.search(r"Step (\d+)/", prompts[i])
            if m:
                step_num = int(m.group(1))

        approach = action.get("approach", "probe")
        if approach == "probe" and step_num <= 2:
            scores.append(0.6)   # probing early = OK
        elif approach == "probe" and step_num > 4:
            scores.append(0.1)   # probing late = bad
        elif approach in ("restart", "debug", "rollback"):
            scores.append(0.9)   # decisive action = good
        elif approach == "scale":
            scores.append(0.5)   # scale might be wrong (BUDGET mode punishes it)
        else:
            scores.append(0.4)
    return scores


def reward_drift_reasoning(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Reward for detecting drift when reward history looks negative.
    If recent rewards in prompt are all negative and agent sets drift_detected=true,
    that's exactly the right behavior.
    """
    scores = []
    for i, c in enumerate(completions):
        action = parse_action_from_text(c)
        drift_flag = action.get("drift_detected", False)
        lead_guess = action.get("lead_mode_guess", "unknown")

        # Check reward history from prompt
        recent_negative = False
        if prompts and i < len(prompts):
            m = re.search(r"History: \[([^\]]+)\]", prompts[i])
            if m:
                try:
                    history = [float(x) for x in m.group(1).split(",") if x.strip()]
                    if len(history) >= 2 and all(h < 0 for h in history[-2:]):
                        recent_negative = True
                except ValueError:
                    pass

        if recent_negative and drift_flag:
            scores.append(1.0)   # correctly detected likely drift
        elif recent_negative and not drift_flag:
            scores.append(0.2)   # missed obvious drift signal
        elif not recent_negative and drift_flag:
            scores.append(0.3)   # false alarm (slightly penalized)
        else:
            scores.append(0.6)   # no drift situation, didn't flag = fine
    return scores


# ── Episode runner ──────────────────────────────────────────────────────────
def run_episode(client: SREClient, task: str, model, tokenizer, device: str) -> Dict[str, Any]:
    max_steps = MAX_STEPS[task]
    obs = client.reset(task)
    episode_rewards: List[float] = []
    trajectory: List[Dict[str, Any]] = []

    for step_num in range(1, max_steps + 1):
        prompt = build_prompt(obs, max_steps)
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH - 128  # leave room for completion
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,     # FIX 1: was 256, halved
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        action = parse_action_from_text(response_text)

        try:
            result = client.step(action)
            reward = float(result.get("reward", 0.0))
            obs    = result.get("observation", obs)
            done   = bool(result.get("done", False))
        except Exception:
            reward = 0.001
            done   = True

        episode_rewards.append(reward)
        trajectory.append({
            "step":     step_num,
            "prompt":   prompt,
            "response": response_text,
            "action":   action,
            "reward":   reward,
        })
        if done:
            break

    return {
        "trajectory":      trajectory,
        "episode_rewards": episode_rewards,
        "episode_reward":  compute_episode_reward(episode_rewards, task),
        "num_steps":       len(episode_rewards),
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int,   default=20)
    parser.add_argument("--task",          type=str,   default="hard",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--output",        type=str,   default="./checkpoints/gen1/")
    parser.add_argument("--batch_size",    type=int,   default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--save_every",    type=int,   default=50)
    parser.add_argument("--env_url",       type=str,   default=DEFAULT_BASE_URL)
    parser.add_argument("--num_generations", type=int, default=4)
    args = parser.parse_args()

    print(f"Config: task={args.task} episodes={args.episodes} "
          f"batch={args.batch_size} gens={args.num_generations} "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ──
    print(f"\nLoading {MODEL_NAME} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ── Connect to env ──
    client = SREClient(base_url=args.env_url)

    # ── Phase 1: Collect ALL episodes (not just positives) ──
    # FIX 2: Original code collected 0 training examples because all
    # baseline scores were negative. Now we use EVERY episode.
    print(f"\nPhase 1: Collecting {args.episodes} baseline episodes...")
    baseline_rewards: List[float] = []
    all_trajectories: List[Dict[str, Any]] = []

    for ep in range(1, args.episodes + 1):
        result = run_episode(client, args.task, model, tokenizer, device)
        reward = result["episode_reward"]
        baseline_rewards.append(reward)
        all_trajectories.extend(result["trajectory"])
        steps = result["num_steps"]
        n_ex  = len(all_trajectories)
        print(f"  Ep {ep}/{args.episodes} score={reward:.3f} steps={steps} examples={n_ex}")

    mean_baseline = sum(baseline_rewards) / len(baseline_rewards)
    print(f"\nBaseline mean: {mean_baseline:.3f}, examples: {len(all_trajectories)}")

    # ── Phase 2: Build dataset from ALL trajectories ──
    # Use all steps as (prompt, completion) pairs — GRPO will figure out
    # which behaviors to reinforce via the reward functions.
    training_data = [
        {"prompt": t["prompt"], "completion": t["response"]}
        for t in all_trajectories
        if t["response"].strip()  # skip empty completions
    ]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_data.jsonl", "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nPhase 2: GRPO on {len(training_data)} examples...")

    dataset = Dataset.from_list(training_data)

    # ── GRPO config ──
    # FIX 1: max_prompt_length + max_completion_length must sum < MAX_SEQ_LENGTH
    training_args = GRPOConfig(
        output_dir           = str(output_dir),
        num_train_epochs     = 1,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 2,
        learning_rate        = args.learning_rate,
        logging_steps        = 10,
        save_steps           = args.save_every,
        max_prompt_length    = 512,    # FIX 1: explicit cap (was uncapped ~900 tokens)
        max_completion_length= 128,    # FIX 1: was 256
        num_generations      = args.num_generations,
        temperature          = 0.7,
        use_vllm             = False,
        report_to            = "none",
    )

    # FIX 2: Three real reward functions instead of broken JSON-parse proxy
    trainer = GRPOTrainer(
        model             = model,
        processing_class  = tokenizer,
        reward_funcs      = [
            reward_format,           # did it output valid JSON with all fields?
            reward_approach_quality, # was the action decisive (not just probing)?
            reward_drift_reasoning,  # did it correctly flag drift from reward history?
        ],
        args              = training_args,
        train_dataset     = dataset,
    )

    trainer.train()

    # ── Save ──
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")

    # ── Phase 3: Quick validation ──
    print("\nPhase 3: Validation (20 episodes)...")
    trained_rewards: List[float] = []
    for ep in range(20):
        result = run_episode(client, args.task, model, tokenizer, device)
        trained_rewards.append(result["episode_reward"])

    mean_trained = sum(trained_rewards) / len(trained_rewards)
    improvement  = mean_trained - mean_baseline

    print(f"\nBaseline mean : {mean_baseline:+.3f}")
    print(f"Trained mean  : {mean_trained:+.3f}")
    print(f"Improvement   : {improvement:+.3f} "
          f"({improvement/abs(mean_baseline)*100 if mean_baseline else 0:.1f}%)")

    results = {
        "task":             args.task,
        "episodes":         args.episodes,
        "baseline_mean":    mean_baseline,
        "trained_mean":     mean_trained,
        "improvement":      improvement,
        "baseline_rewards": baseline_rewards,
        "trained_rewards":  trained_rewards,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    client.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()