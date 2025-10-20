from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .dataset import VOCAB_SIZE


@dataclass(slots=True)
class TinyPhysicsModelConfig:
  state_dim: int
  context: int
  hidden_size: int = 192
  num_layers: int = 2
  dropout: float = 0.1
  token_embed_dim: int = 32
  vocab_size: int = VOCAB_SIZE


class TinyPhysicsNet(nn.Module):
  """
  Autoregressive network that consumes the same inputs as the production TinyPhysics simulator.

  Inputs:
    * states: (batch, context, state_dim), containing steer + road features
    * tokens: (batch, context), integer history of past lateral acceleration tokens
  Output:
    * logits: (batch, vocab_size) distribution over next-step lateral acceleration token
  """

  def __init__(self, config: TinyPhysicsModelConfig) -> None:
    super().__init__()
    self.config = config

    self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.token_embed_dim)

    encoder_input_dim = self.config.state_dim + self.config.token_embed_dim
    gru_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
    self.encoder = nn.GRU(
      input_size=encoder_input_dim,
      hidden_size=self.config.hidden_size,
      num_layers=self.config.num_layers,
      batch_first=True,
      dropout=gru_dropout,
    )
    self.head = nn.Sequential(
      nn.LayerNorm(self.config.hidden_size),
      nn.GELU(),
      nn.Linear(self.config.hidden_size, self.config.hidden_size),
      nn.GELU(),
      nn.Linear(self.config.hidden_size, self.config.vocab_size),
    )

  def forward(self, states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    token_features = self.token_embedding(tokens)
    encoder_in = torch.cat([states, token_features], dim=-1)
    encoded, _ = self.encoder(encoder_in)
    logits = self.head(encoded[:, -1])
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
