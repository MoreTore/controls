from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence
import threading

import pandas as pd
from tqdm import tqdm

from openpilot.tools.lib.logreader import LogReader, ReadMode

ACC_G = 9.81
SEGMENT_LENGTH_NS = int(60 * 1e9)  # one minute per segment
CSV_COLUMNS = ("t", "vEgo", "aEgo", "roll", "targetLateralAcceleration", "steerCommand", "actualLateralAcceleration", "steeringTorque")
MIN_SEGMENT_ROWS = 200


@dataclass(slots=True)
class SegmentSamples:
  route: str
  segment_index: int
  frame: pd.DataFrame


class SegmentWriter:
  def __init__(self, output_path: Path | str):
    self.output_dir = Path(output_path)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._index: list[dict[str, object]] = []
    self._counter = 0
    self._lock = threading.Lock()

  def write(self, segment: SegmentSamples) -> None:
    frame = segment.frame
    if frame.empty:
      return

    with self._lock:
      stem = f"{self._counter:05d}"
      self._counter += 1
      csv_path = self.output_dir / f"{stem}.csv"
      frame.to_csv(csv_path, index=False)
      self._index.append({
        "id": stem,
        "route": segment.route,
        "segment_index": segment.segment_index,
        "num_rows": int(len(frame)),
      })

  def finalize(self) -> None:
    index_path = self.output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
      json.dump(self._index, f, indent=2)


@dataclass(slots=True)
class _Signal:
  timestamp: Optional[float] = None
  value: Optional[float] = None

  def update(self, value: float, timestamp: float) -> None:
    self.value = value
    self.timestamp = timestamp

  def fresh(self, current_time: float, max_age: float) -> bool:
    return self.value is not None and self.timestamp is not None and (current_time - self.timestamp) <= max_age


class TinyPhysicsLogExtractor:
  """
  Extracts TinyPhysics supervision segments from openpilot logs.

  Segments are aligned to the standard 1-minute clip boundaries used by the
  controls challenge dataset and emitted with the same column layout:
    t, vEgo, aEgo, roll, targetLateralAcceleration, steerCommand
  """

  def __init__(self, *, min_speed: float = 0.0, max_signal_age: float = 0.3) -> None:
    self.min_speed = min_speed
    self.max_signal_age = max_signal_age

  def extract_route(self, route_or_segment: str, *, sort_by_time: bool = True) -> List[SegmentSamples]:
    normalized_route = route_or_segment[:-2] if route_or_segment.endswith("/q") else route_or_segment
    buffers = {
      "v_ego": _Signal(),
      "a_ego": _Signal(),
      "roll": _Signal(),
      "steer_command": _Signal(),
      "target_lataccel": _Signal(),
      "actual_lataccel": _Signal(),
      "torque_active": _Signal(),
      "steering_pressed": _Signal(),
      "driver_torque": _Signal(),
    }

    try:
      lr = LogReader(route_or_segment, default_mode=ReadMode.QLOG, sort_by_time=sort_by_time)
    except Exception as e:
      print(e)
      return

    segments: List[SegmentSamples] = []
    base_time_ns: Optional[int] = None
    current_segment_idx: Optional[int] = None
    segment_start_time: Optional[float] = None
    segment_samples: List[dict[str, float]] = []
    torque_active_all: bool = True
    steering_pressed_any: bool = False

    def finalize_segment() -> None:
      nonlocal segment_samples, torque_active_all, steering_pressed_any
      if not segment_samples or not torque_active_all or steering_pressed_any:
        segment_samples = []
        torque_active_all = True
        steering_pressed_any = False
        return

      frame = pd.DataFrame(segment_samples, columns=CSV_COLUMNS)
      if len(frame) < MIN_SEGMENT_ROWS:
        segment_samples = []
        torque_active_all = True
        steering_pressed_any = False
        return
      segments.append(SegmentSamples(
        route=normalized_route,
        segment_index=current_segment_idx or 0,
        frame=frame,
      ))
      segment_samples = []
      torque_active_all = True
      steering_pressed_any = False

    for msg in lr:
      msg_time_ns = msg.logMonoTime
      msg_time = msg_time_ns * 1.0e-9
      which = msg.which()

      if base_time_ns is None:
        base_time_ns = msg_time_ns

      segment_idx = int((msg_time_ns - base_time_ns) // SEGMENT_LENGTH_NS)
      if current_segment_idx is None:
        current_segment_idx = segment_idx
      elif segment_idx != current_segment_idx:
        finalize_segment()
        current_segment_idx = segment_idx
        segment_start_time = None

      if which == "carState":
        car_state = msg.carState
        buffers["v_ego"].update(car_state.vEgo, msg_time)
        buffers["a_ego"].update(car_state.aEgo, msg_time)
        buffers["steering_pressed"].update(float(car_state.steeringPressed), msg_time)
        buffers["driver_torque"].update(car_state.steeringTorque, msg_time)

        if segment_start_time is None:
          segment_start_time = msg_time

        if self._signals_ready(buffers, msg_time):
          if abs(buffers["v_ego"].value or 0.0) < self.min_speed:
            continue

          torque_active_all = torque_active_all and bool(buffers["torque_active"].value)
          steering_pressed_any = steering_pressed_any or bool(buffers["steering_pressed"].value)

          t_rel = msg_time - (segment_start_time or msg_time)
          segment_samples.append({
            "t": float(t_rel),
            "vEgo": float(buffers["v_ego"].value),
            "aEgo": float(buffers["a_ego"].value),
            "roll": float(buffers["roll"].value),
            "targetLateralAcceleration": float(buffers["target_lataccel"].value),
            "steerCommand": float(buffers["steer_command"].value),
            "actualLateralAcceleration": float(buffers["actual_lataccel"].value),
            "steeringTorque": float(buffers["driver_torque"].value),
          })

      elif which == "carControl":
        actuators = msg.carControl.actuators
        steer = actuators.torque if math.isfinite(actuators.torque) else 0.0
        # Dataset convention: steerCommand is left-positive, hence the negation.
        buffers["steer_command"].update(-steer, msg_time)

        if len(msg.carControl.orientationNED) >= 1:
          roll = float(msg.carControl.orientationNED[0])
          buffers["roll"].update(roll, msg_time)

      elif which == "controlsState":
        lateral_state_union = msg.controlsState.lateralControlState
        state_name = lateral_state_union.which()
        lateral_state = getattr(lateral_state_union, state_name)
        if hasattr(lateral_state, "desiredLateralAccel"):
          buffers["target_lataccel"].update(float(lateral_state.desiredLateralAccel), msg_time)
        if state_name == "torqueState":
          if hasattr(lateral_state, "active"):
            buffers["torque_active"].update(float(lateral_state.active), msg_time)
          if hasattr(lateral_state, "actualLateralAccel"):
            buffers["actual_lataccel"].update(float(lateral_state.actualLateralAccel), msg_time)

      elif which == "liveLocationKalman":
        orientation = msg.liveLocationKalman.orientationNED
        if orientation.valid and len(orientation.value) >= 1:
          roll = float(orientation.value[0])
          buffers["roll"].update(roll, msg_time)

    finalize_segment()
    return [segment for segment in segments if not segment.frame.empty]

  def extract_routes(self, routes: Iterable[str], *, show_progress: bool = True) -> List[SegmentSamples]:
    segments: List[SegmentSamples] = []
    iterable = tqdm(list(routes), desc="Extract TinyPhysics segments") if show_progress else routes
    for route in iterable:
      if show_progress:
        tqdm.write(f"Extracting {route} (qlog)...")
      else:
        print(f"Extracting {route} (qlog)...")
      segments.extend(self.extract_route(route))

    return segments

  def _signals_ready(self, buffers: dict[str, _Signal], current_time: float) -> bool:
    required_keys = ("v_ego", "a_ego", "roll", "steer_command", "target_lataccel", "actual_lataccel", "torque_active", "steering_pressed", "driver_torque")
    return all(buffers[key].fresh(current_time, self.max_signal_age) for key in required_keys)


def save_samples(segments: Sequence[SegmentSamples], output_path: str) -> None:
  """Persist extracted segments to disk."""
  writer = SegmentWriter(output_path)
  for segment in segments:
    writer.write(segment)
  writer.finalize()


def load_samples(input_path: str) -> List[SegmentSamples]:
  """
  Load previously extracted segments.

  Expects the directory structure produced by `save_samples`.
  """
  input_dir = Path(input_path)
  if not input_dir.is_dir():
    raise ValueError("samples path must be a directory produced by `save_samples`")

  index_path = input_dir / "index.json"
  segments: List[SegmentSamples] = []

  if index_path.exists():
    with index_path.open("r", encoding="utf-8") as f:
      entries = json.load(f)

    for entry in entries:
      stem = entry["id"]
      frame = pd.read_csv(input_dir / f"{stem}.csv")
      if "actualLateralAcceleration" not in frame.columns:
        continue
      segments.append(SegmentSamples(
        route=entry["route"],
        segment_index=int(entry["segment_index"]),
        frame=frame,
      ))
    if segments:
      return segments

  # Fallback: assume directory contains standalone CSV segments (no metadata)
  csv_files = sorted(input_dir.glob("*.csv"))
  if not csv_files:
    raise FileNotFoundError(f"No segment CSV files found in {input_dir}")

  for csv_path in csv_files:
    frame = pd.read_csv(csv_path)
    if "actualLateralAcceleration" not in frame.columns:
      continue
    segments.append(SegmentSamples(
      route=csv_path.stem,
      segment_index=0,
      frame=frame,
    ))

  if not segments:
    raise FileNotFoundError(f"No segment CSV files found in {input_dir}")

  return segments
