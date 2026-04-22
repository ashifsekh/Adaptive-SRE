import random
from typing import Literal, Optional


class LeadEngineer:
    MODE_SEQUENCE = ["paranoia", "budget", "velocity"]

    def __init__(self, task: str = "hard"):
        self.mode: Optional[str] = None
        self.drift_step: Optional[int] = None
        self.drift_occurred: bool = False
        self.task: str = task

    def reset(self, task: str = "hard") -> None:
        self.task = task
        self.drift_occurred = False

        if task == "easy":
            self.mode = "paranoia"
            self.drift_step = None
        elif task == "medium":
            self.mode = "budget"
            self.drift_step = None
        elif task == "hard":
            self.mode = "paranoia"
            self.drift_step = random.randint(8, 14)
        else:
            self.mode = "paranoia"
            self.drift_step = None

    def check_drift(self, step_number: int) -> bool:
        if self.drift_step is None:
            return False
        if step_number == self.drift_step:
            self.drift_occurred = True
            current_idx = self.MODE_SEQUENCE.index(self.mode)
            next_idx = (current_idx + 1) % len(self.MODE_SEQUENCE)
            self.mode = self.MODE_SEQUENCE[next_idx]
            return True
        return False

    def compute_policy_alignment(self, approach: str, probe_count: int = 0) -> float:
        alignment_rewards = {
            "paranoia": {
                "scale": 0.5,
                "restart": -0.3,
                "debug": 0.2,
                "rollback": 0.3,
                "probe": 0.1
            },
            "budget": {
                "scale": -0.5,
                "restart": 0.4,
                "debug": 0.3,
                "rollback": 0.2,
                "probe": 0.0
            },
            "velocity": {
                "scale": 0.3,
                "restart": 0.4,
                "debug": 0.2,
                "rollback": 0.2,
                "probe": -0.05
            }
        }
        if self.mode not in alignment_rewards:
            return 0.0
        base_reward = alignment_rewards[self.mode].get(approach, 0.0)
        if approach == "probe" and probe_count > 4:
            extra_penalty = (probe_count - 4) * 0.05
            return base_reward - extra_penalty
        return base_reward

    def get_mode_for_observation(self, task: str) -> str:
        if task == "easy":
            return self.mode if self.mode else "paranoia"
        return "unknown"
