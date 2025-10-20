from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from controls_challenge.training.data import SegmentSamples

ACC_G = 9.81
LATACCEL_RANGE = (-5.0, 5.0)
VOCAB_SIZE = 1024


class LataccelTokenizer:
  def __init__(self, *, vocab_size: int = VOCAB_SIZE, value_range: Tuple[float, float] = LATACCEL_RANGE) -> None:
    self.vocab_size = vocab_size
    self.value_range = value_range
    self.bins = np.linspace(self.value_range[0], self.value_range[1], self.vocab_size)

  def encode(self, values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, self.value_range[0], self.value_range[1])
    return np.digitize(clipped, self.bins, right=True)

  def decode(self, tokens: np.ndarray) -> np.ndarray:
    return self.bins[np.clip(tokens, 0, self.vocab_size - 1)]


@dataclass(slots=True)
class TinyPhysicsDatasetConfig:
  context: int = 20
  tokenizer: LataccelTokenizer = field(default_factory=LataccelTokenizer)
  normalize_states: bool = False
  state_mean: Optional[np.ndarray] = None
  state_std: Optional[np.ndarray] = None
  target_mean: Optional[float] = None
  target_std: Optional[float] = None


class TinyPhysicsDataset(Dataset):
  """
  Builds autoregressive training samples from TinyPhysics segments.

  Each dataset item contains:
    * `states`: (context, 4) float32 tensor [steer_action, road_lataccel, v_ego, a_ego]
    * `tokens`: (context,) int64 tensor of discretized historical lateral acceleration
    * `target_token`: int64 of the next-step lateral acceleration token
    * `target_lataccel`: float32 of the next-step actual lateral acceleration
    * `meta`: route, segment index, and timestamp of the prediction target
  """

  def __init__(self, segments: Sequence[SegmentSamples], config: TinyPhysicsDatasetConfig) -> None:
    self.config = config
    self.segments: List[dict[str, np.ndarray | str | int]] = []
    self.indices: List[Tuple[int, int]] = []

    self.state_count = 0
    self.state_mean = np.zeros(4, dtype=np.float32)
    self.state_m2 = np.zeros(4, dtype=np.float32)
    self.target_count = 0
    self.target_mean = 0.0
    self.target_m2 = 0.0

    for segment in segments:
      frame = segment.frame
      if len(frame) <= config.context:
        continue

      steer_action = -frame["steerCommand"].to_numpy(dtype=np.float32)
      roll = frame["roll"].to_numpy(dtype=np.float32)
      road_lataccel = np.sin(roll) * ACC_G
      v_ego = frame["vEgo"].to_numpy(dtype=np.float32)
      a_ego = frame["aEgo"].to_numpy(dtype=np.float32)
      states = np.stack([steer_action, road_lataccel, v_ego, a_ego], axis=1)
      actual_lataccel = frame["actualLateralAcceleration"].to_numpy(dtype=np.float32)
      tokens = self.config.tokenizer.encode(actual_lataccel).astype(np.int64)
      times = frame["t"].to_numpy(dtype=np.float32)

      batch_count = states.shape[0]
      if batch_count <= self.config.context:
        continue

      # Update running statistics
      batch_mean = states.mean(axis=0)
      batch_var = states.var(axis=0)
      delta = batch_mean - self.state_mean
      total = self.state_count + batch_count
      if total > 0:
        self.state_mean += delta * (batch_count / total)
        self.state_m2 += batch_var * batch_count + (delta ** 2) * (self.state_count * batch_count / total)
      self.state_count = total

      target_batch_mean = float(actual_lataccel.mean())
      target_batch_var = float(actual_lataccel.var())
      target_delta = target_batch_mean - self.target_mean
      target_total = self.target_count + batch_count
      if target_total > 0:
        self.target_mean += target_delta * (batch_count / target_total)
        self.target_m2 += target_batch_var * batch_count + (target_delta ** 2) * (self.target_count * batch_count / target_total)
      self.target_count = target_total

      segment_data = {
        "states": states,
        "tokens": tokens,
        "targets": actual_lataccel,
        "times": times,
        "route": segment.route,
        "segment": segment.segment_index,
      }
      window_limit = batch_count - self.config.context
      if window_limit <= 0:
        continue

      segment_id = len(self.segments)
      self.segments.append(segment_data)
      for start in range(window_limit):
        self.indices.append((segment_id, start))

    if not self.indices:
      raise ValueError("No segments contained enough data to create training windows.")

    if self.state_count > 1:
      computed_state_std = np.sqrt(self.state_m2 / (self.state_count - 1))
    else:
      computed_state_std = np.ones(4, dtype=np.float32)
    computed_state_std[computed_state_std < 1e-6] = 1.0

    if self.target_count > 1:
      computed_target_std = float(np.sqrt(self.target_m2 / (self.target_count - 1)))
    else:
      computed_target_std = 1.0
    if computed_target_std < 1e-6:
      computed_target_std = 1.0

    if self.config.state_mean is not None and self.config.state_std is not None:
      self.state_mean = np.asarray(self.config.state_mean, dtype=np.float32)
      self.state_std = np.asarray(self.config.state_std, dtype=np.float32)
      self.state_std[self.state_std < 1e-6] = 1.0
    else:
      self.state_mean = self.state_mean.astype(np.float32)
      self.state_std = computed_state_std.astype(np.float32)

    if self.config.target_mean is not None and self.config.target_std is not None:
      self.target_mean = float(self.config.target_mean)
      self.target_std = float(max(self.config.target_std, 1e-6))
    else:
      self.target_mean = float(self.target_mean)
      self.target_std = float(computed_target_std)

    if self.config.normalize_states:
      for segment_data in self.segments:
        segment_data["states"] = ((segment_data["states"] - self.state_mean) / self.state_std).astype(np.float32)
    else:
      for segment_data in self.segments:
        segment_data["states"] = segment_data["states"].astype(np.float32)

    self.state_mean = self.state_mean.astype(np.float32)
    self.state_std = self.state_std.astype(np.float32)
    self.target_mean = float(self.target_mean)
    self.target_std = float(self.target_std)

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict[str, object]]:
    segment_id, start = self.indices[idx]
    seg = self.segments[segment_id]

    end = start + self.config.context
    target_idx = end

    states = torch.from_numpy(seg["states"][start:end])
    token_context = torch.from_numpy(seg["tokens"][start:end])
    target_token = torch.tensor(seg["tokens"][target_idx], dtype=torch.long)
    target_lataccel = torch.tensor(seg["targets"][target_idx], dtype=torch.float32)

    meta: Dict[str, object] = {
      "route": seg["route"],
      "segment": int(seg["segment"]),
      "t": float(seg["times"][target_idx]),
    }

    return {
      "states": states,
      "tokens": token_context,
      "target_token": target_token,
      "target_lataccel": target_lataccel,
      "meta": meta,
    }


def train_val_split(segments: Sequence[SegmentSamples], *, val_pct: float = 0.2) -> tuple[List[SegmentSamples], List[SegmentSamples]]:
  if not (0.0 < val_pct < 1.0):
    raise ValueError("val_pct must be within (0, 1)")

  routes = sorted({segment.route for segment in segments})
  if not routes:
    return [], []

  cutoff = max(1, int(len(routes) * (1.0 - val_pct)))
  train_routes = set(routes[:cutoff])

  train_segments = [segment for segment in segments if segment.route in train_routes]
  val_segments = [segment for segment in segments if segment.route not in train_routes]
  return train_segments, val_segments
