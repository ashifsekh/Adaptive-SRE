# PROGRESS.md — AdaptiveSRE Build Status

Last updated: 2026-04-26
Current phase: 10

## Completed phases

- [x] Phase 0 — Init
- [x] Phase 1 — Mock services
- [x] Phase 2 — Models + service graph
- [x] Phase 3 — Lead engineer + fault injector + docker executor
- [x] Phase 4 — Grader
- [x] Phase 5 — Environment core
- [x] Phase 6 — FastAPI server + Gradio UI
- [x] Phase 7 — inference.py
- [x] Phase 8 — openenv.yaml + Dockerfile
- [x] Phase 9 — Training pipeline
- [ ] Phase 10 — Full validation

## Files created (fill as built)

- AGENT.md
- MASTER_BUILD_GUIDE.md
- requirements.txt
- mock_services/db/main.py, Dockerfile
- mock_services/auth/main.py, Dockerfile
- mock_services/payment/main.py, Dockerfile
- mock_services/cache/main.py, Dockerfile
- mock_services/notification/main.py, Dockerfile
- mock_services/docker-compose.yml
- server/**init**.py
- server/models.py
- server/service_graph.py
- server/lead_engineer.py
- server/docker_executor.py
- server/fault_injector.py
- server/grader.py
- server/environment.py
- server/app.py
- inference.py
- openenv.yaml
- Dockerfile

## Decisions that deviate from AGENT.md

- DB port changed from 5432 to 15432 (local PostgreSQL uses 5432)

## Critical bugfix updates (2026-04-25)

- requirements.txt: trl version bumped to >=0.18.2,<=0.24.0
- train.py: fixed training-pipeline issues by removing dead reward wrapper code and duplicate GRPO trainer initialization
- train_colab.ipynb: fixed uvicorn host binding in Cell 2 startup subprocesses
- eval.py: added direct mode support for Colab (env_url=direct)
- Phase 9 status: Code complete, pending Colab validation run

## Measured results (ACTUAL)

Run 1 (easy, CPU, 1B, episodes=8, gens=2):

- Gen 0 mean reward: -0.193
- Gen 1 mean reward: -0.167
- Improvement: +0.026
- GRPO training time: 1:20:05

Run 2 (easy, CPU, 1B, episodes=8, gens=2):

- Gen 0 mean reward: -0.197
- Gen 1 mean reward: -0.167
- Improvement: +0.029
- GRPO training time: 1:13:54

Summary across 2 runs:

- Gen 0 mean reward (avg): -0.195
- Gen 1 mean reward (avg): -0.167
- Improvement (avg): +0.028 (+14.4%)
- Training time: ~75-80 min on CPU
- Runs completed: 2 (reproducible)

Gen 0 hard task baseline: TBD

## Next step

Phase 10 — Full validation: Run Colab validation and record measured training/eval results
