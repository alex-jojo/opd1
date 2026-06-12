"""TA-OPD trainer entrypoint.

The TA-OPD-specific data path is intentionally kept in the TA-OPD actor so this
trainer can reuse the upstream PPO control flow without touching it.
"""

import time

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RayTAOPDTrainer(RayPPOTrainer):
    """PPO trainer with TA-OPD worker/actor wiring supplied by main_ta_opd."""

    def _debug_progress(self, message: str) -> None:
        if not self._progress_debug_enabled():
            return
        total_steps = getattr(self, "total_training_steps", "?")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ta_opd_progress] {timestamp} step={self.global_steps}/{total_steps} {message}", flush=True)
