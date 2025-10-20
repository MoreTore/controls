from __future__ import annotations

import argparse
import json
import math
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpilot.tools.lib.api import APIError, CommaApi, UnauthorizedError
from openpilot.tools.lib.auth_config import get_token

from controls_challenge.training.data import SegmentSamples, SegmentWriter, TinyPhysicsLogExtractor, load_samples
from controls_challenge.training.dataset import TinyPhysicsDataset, TinyPhysicsDatasetConfig, train_val_split
from controls_challenge.training.model import TinyPhysicsModelConfig, TinyPhysicsNet, export_to_onnx


def generate_debug_report(segments: List[SegmentSamples], report_path: Path, max_segments: int = 100) -> None:
  if not segments:
    return

  report_path.parent.mkdir(parents=True, exist_ok=True)
  html_lines = [
    "<html><head><meta charset='utf-8'><title>TinyPhysics Segment Debug Report</title>",
    "<style>body{font-family:Arial,sans-serif;background:#f7f7f7;color:#222;}h2{margin-top:40px;}figure{background:#fff;padding:12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);}</style>",
    "</head><body><h1>TinyPhysics Segment Diagnostics</h1>",
    f"<p>Segments plotted: {min(len(segments), max_segments)} / {len(segments)}</p>",
  ]

  from io import BytesIO
  import base64
  import matplotlib.pyplot as plt

  for idx, segment in enumerate(segments[:max_segments]):
    frame = segment.frame
    if frame.empty:
      continue
    t = frame["t"].to_numpy()
    target = frame["targetLateralAcceleration"].to_numpy()
    actual = frame.get("actualLateralAcceleration", target).to_numpy()
    steer = frame["steerCommand"].to_numpy()
    v_ego = frame["vEgo"].to_numpy()
    roll = frame["roll"].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, target, label="Target Lat Accel", color="#1f77b4")
    axes[0].plot(t, actual, label="Actual Lat Accel", color="#ff7f0e", alpha=0.8)
    axes[0].set_ylabel("m/s²")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, steer, color="#2ca02c")
    axes[1].set_ylabel("Steer Cmd")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, v_ego, color="#d62728")
    axes[2].set_ylabel("vEgo (m/s)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, roll, color="#9467bd")
    axes[3].set_ylabel("Roll (rad)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    html_lines.append(f"<figure><h2>Segment {idx:05d} | Route: {segment.route}/{segment.segment_index}</h2>")
    html_lines.append(f"<img src='data:image/png;base64,{encoded}' alt='Segment {idx:05d} plot' style='width:100%;height:auto;'/>")
    html_lines.append("</figure>")

  html_lines.append("</body></html>")
  report_path.write_text("\n".join(html_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train a TinyPhysics surrogate model from openpilot logs.")
  parser.add_argument("--routes", nargs="+", default=[], help="Route or segment identifiers accepted by LogReader.")
  parser.add_argument("--routes-file", type=str, help="Optional newline-delimited list of routes.")
  parser.add_argument("--dongle-id", action="append", dest="dongle_ids", default=[], help="Fetch all routes for these dongle IDs via the comma API.")
  parser.add_argument("--routes-limit", type=int, default=100, help="Maximum routes to fetch per dongle (0 for no limit).")
  parser.add_argument("--git-commit", action="append", dest="git_commits", default=[], help="Only include routes whose git_commit matches (case-insensitive).")
  parser.add_argument("--min-speed", type=float, default=5.0, help="Discard samples below this ego speed (m/s).")
  parser.add_argument("--samples-path", type=str, help="Optional directory cache of extracted segments.")
  parser.add_argument("--write-samples", action="store_true", help="Persist extracted samples to --samples-path after loading.")
  parser.add_argument("--extract-threads", type=int, default=1, help="Number of threads to use for route extraction (>=1).")
  parser.add_argument("--context", type=int, default=20, help="History length used for autoregressive context.")
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--min-lr", type=float, default=3e-5)
  parser.add_argument("--scheduler-warmup", type=int, default=2, help="Number of warmup epochs before cosine decay.")
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--grad-clip", type=float, default=1.0)
  parser.add_argument("--val-pct", type=float, default=0.2, help="Validation split by route.")
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--output-dir", type=str, default="./training_runs/tinyphysics")
  parser.add_argument("--export-onnx", action="store_true", help="Export the trained model to ONNX.")
  parser.add_argument("--onnx-path", type=str, default="./models/tinyphysics_trained.onnx")
  parser.add_argument("--onnx-opset", type=int, default=18, help="ONNX opset version to target (>=18 recommended).")
  parser.add_argument("--seed", type=int, default=2024)
  parser.add_argument("--debug", action="store_true", help="Generate an HTML report plotting each extracted segment.")
  parser.add_argument("--label-smoothing", type=float, default=0.05)
  parser.add_argument("--regression-weight", type=float, default=0.3, help="Weight for auxiliary regression loss.")
  return parser.parse_args()


def normalize_route_identifier(identifier: str) -> str:
  identifier = identifier.strip().rstrip("/")
  if identifier.endswith(("/q", "/r", "/a", "/i")):
    return identifier
  return identifier + "/q"


def fetch_routes_for_dongles(dongle_ids: List[str], *, limit: Optional[int], git_commits: List[str]) -> List[str]:
  token = get_token()
  if token is None:
    raise RuntimeError(
      "No API token found. Authenticate with `python tools/lib/auth.py` or set one via openpilot.tools.lib.auth_config.set_token()."
    )

  api = CommaApi(token)
  commit_filter = {c.lower() for c in git_commits} if git_commits else None
  fetched_routes: List[str] = []

  for dongle in dongle_ids:
    print(f"Fetching routes for dongle {dongle}...")
    remaining = limit if limit and limit > 0 else None
    offset = 0
    while True:
      request_limit = min(100, remaining) if remaining is not None else 100
      if remaining is not None and request_limit <= 0:
        break

      params = {"limit": request_limit, "offset": offset}
      try:
        routes = api.request("GET", f"/v1/devices/{dongle}/routes_segments", params=params)
      except UnauthorizedError as exc:
        raise RuntimeError("Unauthorized when contacting the comma API. Run `python tools/lib/auth.py` to authenticate.") from exc
      except APIError as exc:
        raise RuntimeError(f"Failed to fetch routes for dongle {dongle}: {exc}") from exc

      if not routes:
        break

      for route in routes:
        fullname = route.get("fullname")
        if not fullname:
          continue
        commit = (route.get("git_commit") or "").lower()
        if commit_filter and commit not in commit_filter:
          continue
        if route.get("maxqlog", 0) <= 0:
          continue

        print(f"  - {fullname} (git_commit={route.get('git_commit')}, segments={len(route.get('segment_numbers', []))})")
        fetched_routes.append(fullname)
        if remaining is not None:
          remaining -= 1
          if remaining <= 0:
            break

      if remaining is not None and remaining <= 0:
        break

      offset += len(routes)
      if len(routes) < request_limit:
        break

  return fetched_routes


def gather_routes(args: argparse.Namespace) -> List[str]:
  routes: List[str] = []
  if args.routes:
    routes.extend(args.routes)
  if args.routes_file:
    with open(args.routes_file, "r", encoding="utf-8") as f:
      for line in f:
        cleaned = line.strip()
        if cleaned:
          routes.append(cleaned)
  if args.dongle_ids:
    limit = None if args.routes_limit == 0 else args.routes_limit
    routes.extend(fetch_routes_for_dongles(args.dongle_ids, limit=limit, git_commits=args.git_commits))
  unique_routes = sorted(set(routes))
  return unique_routes


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: TinyPhysicsNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    token_centers: torch.Tensor,
    regression_weight: float,
    label_smoothing: float,
) -> Dict[str, float]:
  model.train()
  total_loss = 0.0
  total_correct = 0
  total_examples = 0

  for batch in tqdm(loader, desc="train", leave=False):
    batch.pop("meta", None)
    states = batch["states"].to(device)
    tokens = batch["tokens"].to(device)
    targets = batch["target_token"].to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(states, tokens)
    ce_loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    if regression_weight > 0.0:
      probs = F.softmax(logits, dim=-1)
      expected_lataccel = torch.sum(probs * token_centers, dim=-1)
      regression_loss = F.mse_loss(expected_lataccel, batch["target_lataccel"].to(device))
      loss = ce_loss + regression_weight * regression_loss
    else:
      loss = ce_loss
    loss.backward()
    if grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
      scheduler.step()

    total_loss += loss.item() * len(states)
    total_correct += (logits.argmax(dim=-1) == targets).sum().item()
    total_examples += len(states)

  avg_loss = total_loss / max(total_examples, 1)
  accuracy = total_correct / max(total_examples, 1)
  return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def evaluate_epoch(
    model: TinyPhysicsNet,
    loader: DataLoader,
    device: torch.device,
    token_centers: Optional[torch.Tensor] = None,
    regression_weight: float = 0.0,
) -> Dict[str, float]:
  model.eval()
  total_loss = 0.0
  total_correct = 0
  total_examples = 0

  for batch in tqdm(loader, desc="eval", leave=False):
    batch.pop("meta", None)
    states = batch["states"].to(device)
    tokens = batch["tokens"].to(device)
    targets = batch["target_token"].to(device)

    logits = model(states, tokens)
    ce_loss = F.cross_entropy(logits, targets)
    if token_centers is not None and regression_weight > 0.0:
      probs = F.softmax(logits, dim=-1)
      expected_lataccel = torch.sum(probs * token_centers, dim=-1)
      regression_loss = F.mse_loss(expected_lataccel, batch["target_lataccel"].to(device))
      loss = ce_loss + regression_weight * regression_loss
    else:
      loss = ce_loss

    total_loss += loss.item() * len(states)
    total_correct += (logits.argmax(dim=-1) == targets).sum().item()
    total_examples += len(states)

  avg_loss = total_loss / max(total_examples, 1)
  accuracy = total_correct / max(total_examples, 1)
  return {"loss": avg_loss, "accuracy": accuracy}


def main() -> None:
  args = parse_args()
  set_seed(args.seed)
  device = torch.device(args.device)

  output_dir = Path(args.output_dir).expanduser()
  output_dir.mkdir(parents=True, exist_ok=True)

  samples_path = Path(args.samples_path).expanduser() if args.samples_path else None
  if samples_path and samples_path.suffix:
    raise ValueError("--samples-path must point to a directory (no file extension).")

  samples: List[SegmentSamples] | None = None

  if samples_path and samples_path.exists() and not args.write_samples:
    if not samples_path.is_dir():
      raise ValueError("--samples-path must point to a directory when caching segments.")
    print(f"Loading cached samples from {samples_path}")
    samples = load_samples(str(samples_path))
  else:
    routes = gather_routes(args)
    if not routes:
      raise ValueError("No routes provided; supply --routes or --routes-file.")

    route_identifiers = [normalize_route_identifier(r) for r in routes]

    writer: Optional[SegmentWriter] = None
    if samples_path and args.write_samples:
      if samples_path.exists():
        shutil.rmtree(samples_path)
      writer = SegmentWriter(samples_path)

    collected_segments: List[SegmentSamples] = [] if writer is None else []

    def handle_segments(segment_list: List[SegmentSamples]) -> None:
      if writer is not None:
        for segment in segment_list:
          writer.write(segment)
      else:
        collected_segments.extend(segment_list)

    extract_threads = max(1, args.extract_threads)
    if extract_threads > 1 and len(route_identifiers) > 1:
      extractor_kwargs = {"min_speed": args.min_speed}

      def _process(route: str) -> List[SegmentSamples]:
        local_extractor = TinyPhysicsLogExtractor(**extractor_kwargs)
        return local_extractor.extract_route(route)

      with ThreadPoolExecutor(max_workers=min(extract_threads, len(route_identifiers))) as executor:
        futures = {executor.submit(_process, route): route for route in route_identifiers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extract routes"):
          segments_for_route = future.result()
          handle_segments(segments_for_route)
    else:
      extractor = TinyPhysicsLogExtractor(min_speed=args.min_speed)
      for route in tqdm(route_identifiers, desc="Extract routes"):
        segments_for_route = extractor.extract_route(route)
        handle_segments(segments_for_route)

    if writer is not None:
      writer.finalize()
      samples = load_samples(str(samples_path))
    else:
      samples = collected_segments

  if not samples:
    raise RuntimeError("No samples were extracted from the provided routes.")

  if args.debug:
    report_base = samples_path if samples_path else output_dir / "debug_reports"
    report_path = Path(report_base) / "segments_debug.html"
    generate_debug_report(samples, report_path)
    print(f"Debug report written to {report_path}")

  dataset_cfg = TinyPhysicsDatasetConfig(context=args.context, normalize_states=False)
  train_segments, val_segments = train_val_split(samples, val_pct=args.val_pct)
  if not train_segments:
    raise RuntimeError("Training split is empty; ensure routes were extracted correctly.")

  train_dataset = TinyPhysicsDataset(train_segments, dataset_cfg)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

  print(
    f"Training segments: {len(train_dataset.segments)} | windows: {len(train_dataset)} | "
    f"state_std: {train_dataset.state_std.tolist()} | target_std: {train_dataset.target_std:.4f}"
  )

  token_centers = torch.from_numpy(train_dataset.config.tokenizer.bins.astype(np.float32)).to(device).view(1, -1)

  val_loader: Optional[DataLoader] = None
  if val_segments:
    val_cfg = TinyPhysicsDatasetConfig(
      context=dataset_cfg.context,
      tokenizer=dataset_cfg.tokenizer,
      normalize_states=False,
      state_mean=train_dataset.state_mean,
      state_std=train_dataset.state_std,
      target_mean=train_dataset.target_mean,
      target_std=train_dataset.target_std,
    )
    val_dataset = TinyPhysicsDataset(val_segments, val_cfg)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

  state_dim = train_dataset.segments[0]["states"].shape[1]
  model_cfg = TinyPhysicsModelConfig(
    state_dim=state_dim,
    context=dataset_cfg.context,
    vocab_size=dataset_cfg.tokenizer.vocab_size,
    state_mean=tuple(float(x) for x in train_dataset.state_mean.tolist()),
    state_std=tuple(float(x) for x in train_dataset.state_std.tolist()),
    target_mean=float(train_dataset.target_mean),
    target_std=float(train_dataset.target_std),
  )
  model = TinyPhysicsNet(model_cfg).to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
  total_steps = len(train_loader) * args.epochs
  warmup_steps = int(args.scheduler_warmup * len(train_loader))
  min_lr = args.min_lr
  def lr_lambda(step: int) -> float:
    if step < warmup_steps:
      return max(1e-8, (step + 1) / max(1, warmup_steps))
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return max(min_lr / args.lr, cosine)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

  history: List[Dict[str, float]] = []
  best_val_loss = math.inf
  best_epoch = -1

  for epoch in range(1, args.epochs + 1):
    print(f"\nEpoch {epoch}/{args.epochs}")
    train_stats = train_epoch(
      model,
      train_loader,
      optimizer,
      device,
      args.grad_clip,
      scheduler,
      token_centers,
      args.regression_weight,
      args.label_smoothing,
    )

    stats_record = {"epoch": epoch, "train_loss": train_stats["loss"], "train_accuracy": train_stats["accuracy"]}

    if val_loader is not None:
      val_stats = evaluate_epoch(
        model,
        val_loader,
        device,
        token_centers,
        args.regression_weight,
      )
      stats_record.update({"val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"]})
      if val_stats["loss"] < best_val_loss:
        best_val_loss = val_stats["loss"]
        best_epoch = epoch
        torch.save(model.state_dict(), output_dir / "best_model.pt")
    else:
      torch.save(model.state_dict(), output_dir / "best_model.pt")

    history.append(stats_record)
    print(json.dumps(stats_record, indent=2))

  torch.save(
    {
      "model_state": model.state_dict(),
      "model_config": asdict(model_cfg),
      "dataset_config": {
        "context": dataset_cfg.context,
        "tokenizer_vocab_size": dataset_cfg.tokenizer.vocab_size,
        "tokenizer_range": dataset_cfg.tokenizer.value_range,
        "normalize_states": dataset_cfg.normalize_states,
      },
      "dataset_stats": {
        "state_mean": train_dataset.state_mean.tolist(),
        "state_std": train_dataset.state_std.tolist(),
        "target_mean": train_dataset.target_mean,
        "target_std": train_dataset.target_std,
      },
      "history": history,
      "best_epoch": best_epoch,
    },
    output_dir / "training_artifacts.pt",
  )

  with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

  if args.export_onnx:
    onnx_path = Path(args.onnx_path).expanduser()
    export_to_onnx(
      model,
      output_path=onnx_path,
      context_length=model_cfg.context,
      state_dim=model_cfg.state_dim,
      vocab_size=model_cfg.vocab_size,
      opset=args.onnx_opset,
    )
    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
  main()
