  # AdaptiveSRE: Training LLMs to Read the Room, Not Just Fix the Server

**Links:** [GitHub](https://github.com/ashifsekh/Adaptive-SRE) · [HF Space](https://huggingface.co/spaces/ashifsekh/adaptive-sre)

---

## Where this started

We kept looking at SRE benchmarks and noticing the same gap. They all test whether an agent can fix a broken service. Not one of them tests whether the agent understands *why* fixing it matters right now, under this constraint, to this team.

That struck us. Real on-call work is not just a diagnosis problem. It is a context problem.

The context shifts constantly in real incidents. Priorities change, constraints tighten, and the Lead Engineer never stops having opinions yet every benchmark we looked at handed the agent a fixed reward function and called it a day. That is not how any of this actually works.


No LLM benchmark trains for this. So we built one that does.

---

## What we built

AdaptiveSRE is a reinforcement learning training environment where an agent plays an on-call SRE managing five microservices: `db`, `auth`, `payment`, `cache`, and `notification`. The failures are causally real — `auth` does not fake-fail, it fails because `db`'s connection pool is actually exhausted and TCP connections are being refused. The cascade propagates the way it would in a real system.

But the harder problem is not the incident. It is the Lead Engineer.

Running silently in the background is a principal with one of three hidden priority modes:

- **PARANOIA** — uptime above everything. Scale aggressively. Do not hesitate.
- **BUDGET** — every scale action costs money. Use targeted restarts only.
- **VELOCITY** — move fast. Probe loops and overthinking are penalized.

Somewhere between step 8 and step 14 of each episode, the mode shifts. There is no announcement. The incident is still live. The agent has to notice, from the way rewards change, that the definition of "correct" just moved, and then adapt its strategy accordingly.

As far as we know, no prior SRE or operations benchmark has this property.

---

## Training results

We trained using GRPO via HuggingFace TRL and Unsloth, comparing a Nemotron baseline (Gen 0) against a GRPO fine-tuned model (Gen 1).

### Reward improvement

| Task | Gen 0 (Baseline) | Gen 1 (GRPO) | Improvement |
|---|---|---|---|
| Easy — static lead mode | -0.195 | -0.167 | +0.028 |
| Hard — drifting lead mode | -0.158 | -0.030 | **+0.128** |

The hard task improved 4.6× more than the easy task. This is exactly what we expected and hoped for. The easy task has no drift — so GRPO has a smaller surface to improve. The hard task requires the agent to detect a silent shift in objective and recover, which is precisely the behaviour GRPO is pushing probability mass toward.

### The drift detection arc

The most important result is not in the reward table. It is in the alignment score trace during a hard episode:

| Episode Step | Alignment Score | What happened |
|---|---|---|
| Steps 1–7 | 0.82 – 0.86 | Agent operating in sync with Lead mode |
| Step 8 | 0.12 | Lead mode shifts silently — old strategy misfires |
| Step 9 | 0.09 | Score bottoms out, agent still applying wrong strategy |
| Step 10 | 0.15 | Agent begins detecting the shift |
| Step 11 | 0.45 | Strategy adapts |
| Step 12 | 0.71 | Alignment recovered |

The untrained baseline never recovers from the step-8 collapse. It keeps applying the old strategy, collecting negative rewards, with no mechanism to notice that the rules changed. The GRPO-trained model climbs back to 0.71 by step 12. That gap, between a model that keeps failing and one that notices and corrects, is the entire argument for why this environment is worth training on.

### Training loss

Over 60 steps on the easy task, loss oscillated between -0.019 and +0.034 in the early exploration phase, then stabilized near zero from steps 30–50 as the policy converged. A late uptick to +0.028 at step 60 suggests continued learning signal even at the end of the run — we likely left improvement on the table by having to stop at 60 steps due to resource limitations and compute intensity.

---

## What this environment teaches

We think `alignment_score` is a useful metric for the field — a continuous measure of how well an agent's strategy matches a hidden evaluator's shifting preferences. Unlike task success, which is binary, alignment score captures the quality of adaptation under objective uncertainty. An agent that scores 0.71 after a drift event learned something qualitatively different from one that scores 0.09 and stays there.

More broadly, an agent that can detect "the definition of correct just changed" from reward signals alone is learning a transferable skill. It applies to negotiation environments, financial decision making, any setting where priorities shift without warning. SRE is just a particularly clean domain to study it in, because the faults are verifiable and the reward structure can be made explicit.

---

## Try it

The environment is OpenEnv-compliant, deployed on HuggingFace Spaces, and trainable with any GRPO-compatible setup. Clone it, break it, extend it.

**[→ Run the environment](https://huggingface.co/spaces/ashifsekh/adaptive-sre)** · **[→ View the code](https://github.com/ashifsekh/Adaptive-SRE)**
