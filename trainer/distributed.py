"""Distributed training manager: coordinates multiple Actor processes with a single Learner.

Architecture: Data-parallel — N Actor processes collect experience,
1 Learner process trains the model on GPU. Parameters are synced
via shared memory or PyTorch RPC.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Optional, List
from multiprocessing import Process, Queue, Event, shared_memory
from threading import Thread

from .actor_worker import ActorWorker


class DistributedTrainer:
    """Manages distributed RL training with multiple Actor processes."""

    def __init__(self, agent, env_creator, num_actors: int = 64,
                 sync_interval: int = 100, sample_queue_size: int = 10000,
                 device: str = "cuda"):
        self.agent = agent
        self.env_creator = env_creator
        self.num_actors = num_actors
        self.sync_interval = sync_interval
        self.device = device

        # Communication channels.
        self.sample_queue = Queue(maxsize=sample_queue_size)
        self.param_event = Event()   # signals new params available
        self.stop_event = Event()

        # Shared model parameters (on CPU for Actor access).
        self._shared_params: Dict[str, torch.Tensor] = {}

        # Actor processes.
        self.actors: List[Process] = []
        self._training_thread: Optional[Thread] = None

    def start(self):
        """Launch all Actor processes."""
        print(f"[Distributed] Starting {self.num_actors} actors...")

        # Copy initial params to CPU for sharing.
        self._update_shared_params()

        for actor_id in range(self.num_actors):
            p = Process(
                target=ActorWorker.run,
                args=(
                    actor_id, self.env_creator,
                    self.sample_queue, self.param_event, self.stop_event,
                    self.sync_interval,
                ),
                name=f"Actor-{actor_id}",
                daemon=True,
            )
            p.start()
            self.actors.append(p)

        print(f"[Distributed] All {self.num_actors} actors started.")

    def stop(self):
        """Gracefully stop all actors."""
        self.stop_event.set()
        for p in self.actors:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        print("[Distributed] All actors stopped.")

    def get_samples(self, max_samples: int = 256) -> List[Dict]:
        """Collect samples from the queue (non-blocking)."""
        samples = []
        for _ in range(max_samples):
            try:
                sample = self.sample_queue.get_nowait()
                samples.append(sample)
            except:
                break
        return samples

    def sync_params(self):
        """Push latest model params to shared memory so actors can pull them."""
        self._update_shared_params()
        self.param_event.set()
        self.param_event.clear()

    def _update_shared_params(self):
        """Copy agent params to CPU for sharing."""
        net = getattr(self.agent, 'online_net', None) or getattr(self.agent, 'network', None)
        for name, param in net.state_dict().items():
            self._shared_params[name] = param.cpu().clone()

    def get_shared_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._shared_params.items()}


class ReplayFeeder(Thread):
    """Background thread that moves samples from queue into replay buffer."""

    def __init__(self, sample_queue: Queue, replay_buffer, batch_size: int = 256):
        super().__init__(daemon=True)
        self.queue = sample_queue
        self.buffer = replay_buffer
        self.batch_size = batch_size
        self.running = False
        self.total_fed = 0

    def run(self):
        self.running = True
        while self.running:
            try:
                sample = self.queue.get(timeout=1.0)
                self.buffer.add(
                    sample["state"], sample["action"], sample["reward"],
                    sample["next_state"], sample["done"],
                    td_error=sample.get("td_error"),
                )
                self.total_fed += 1
            except:
                pass

    def stop(self):
        self.running = False
