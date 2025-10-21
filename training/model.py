from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import onnx
import torch
import torch.nn as nn
from onnx2torch import convert


@dataclass(slots=True)
class TinyPhysicsModelConfig:
  state_dim: int
  context: int
  template_path: str = "controls_challenge/models/tinyphysics.onnx"
  load_pretrained: bool = True
  trainable_initializers: bool = True


class TinyPhysicsNet(nn.Module):
  """
  Wrapper around the production TinyPhysics transformer architecture converted
  from the reference ONNX graph. Converting and promoting the initializers to
  parameters keeps the structure identical while allowing fine-tuning.
  """

  def __init__(self, config: TinyPhysicsModelConfig) -> None:
    super().__init__()
    self.config = config

    candidate_paths = [Path(self.config.template_path), Path("controls_challenge/models/tinyphysics.onnx"), Path("models/tinyphysics.onnx")]
    onnx_path: Optional[Path] = None
    for candidate in candidate_paths:
      if candidate.exists():
        onnx_path = candidate
        break
    if onnx_path is None:
      raise FileNotFoundError(f"TinyPhysics template ONNX not found. Tried: {candidate_paths}")

    onnx_model = onnx.load(onnx_path)
    self.backbone = convert(onnx_model)

    if self.config.trainable_initializers:
      self._promote_initializers(self.backbone)

    if not self.config.load_pretrained:
      self._reset_parameters()

  @staticmethod
  def _promote_initializers(module: nn.Module) -> None:
    for submodule in module.modules():
      if not hasattr(submodule, "_buffers"):
        continue
      buffer_names = [name for name in submodule._buffers.keys() if name.startswith("onnx_initializer")]
      for name in buffer_names:
        buf = submodule._buffers.pop(name)
        if not isinstance(buf, torch.Tensor):
          continue
        if buf.is_floating_point() or buf.is_complex():
          param = nn.Parameter(buf.clone().detach())
          submodule.register_parameter(name, param)
        else:
          # Keep non-floating initializers as buffers (indices, masks, etc.)
          submodule.register_buffer(name, buf)

  def _reset_parameters(self) -> None:
    for name, param in self.backbone.named_parameters():
      if param.dim() >= 2:
        nn.init.xavier_uniform_(param)
      else:
        nn.init.zeros_(param)

  def forward(self, states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    outputs = self.backbone(states, tokens)
    if isinstance(outputs, dict):
      logits = outputs.get("output")
    elif isinstance(outputs, (tuple, list)):
      logits = outputs[0]
    else:
      logits = outputs
    if logits.dim() == 3:
      logits = logits[:, -1, :]
    return logits


def export_to_onnx(
    model: nn.Module,
    *,
    output_path: str | Path,
    context_length: int,
    state_dim: int,
    vocab_size: int,
    opset: int = 18,
) -> None:
  """
  Serialize the TinyPhysicsNet to an ONNX graph with the interface expected by tinyphysics.py.
  """
  model.eval()
  dummy_states = torch.zeros(1, context_length, state_dim, dtype=torch.float32)
  dummy_tokens = torch.zeros(1, context_length, dtype=torch.long)

  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  first_param = next(model.parameters(), None)
  orig_device = first_param.device if first_param is not None else torch.device("cpu")
  moved_to_cpu = False
  if orig_device.type != "cpu":
    model.to("cpu")
    moved_to_cpu = True

  class _ExportWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
      super().__init__()
      self.base_model = base_model

    def forward(self, states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
      logits = self.base_model(states, tokens)
      if logits.ndim == 2:
        logits = logits.unsqueeze(1)
      return logits

  export_module = _ExportWrapper(model)

  try:
    torch.onnx.export(
      export_module,
      (dummy_states, dummy_tokens),
      output_path,
      input_names=["states", "tokens"],
      output_names=["logits"],
      opset_version=opset,
      dynamic_axes={"states": {0: "batch"}, "tokens": {0: "batch"}, "logits": {0: "batch"}},
      do_constant_folding=False,
      optimize=False,
      dynamo=False,
    )
  except ModuleNotFoundError as exc:
    missing_mod = exc.name or "onnxscript"
    raise RuntimeError(
      f"ONNX export failed because the optional dependency '{missing_mod}' is not installed. "
      "Install it with `uv pip install onnxscript` (or `pip install onnxscript`) and retry, "
      "or rerun training without `--export-onnx`."
    ) from exc
  except Exception as exc:
    raise RuntimeError(
      "torch.onnx.export failed. Try upgrading `torch`, `onnx`, or `onnxscript`, "
      "or rerun training without `--export-onnx`. "
      f"Original error: {exc}"
    ) from exc
  finally:
    if moved_to_cpu:
      model.to(orig_device)
