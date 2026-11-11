"""Quantize a PyTorch checkpoint and write its tensors to a C header."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import torch


DEFAULT_CHECKPOINT = Path(__file__).with_name("rat_15105_channel_reductiontiny.pt")
DEFAULT_HEADER = DEFAULT_CHECKPOINT.parents[1] / "weights.h"


def describe(value: Any, name: str = "checkpoint", indent: int = 0) -> None:
	"""Print a compact recursive description of a checkpoint value."""
	prefix = " " * indent

	if isinstance(value, torch.Tensor):
		print(
			f"{prefix}{name}: Tensor(shape={tuple(value.shape)}, "
			f"dtype={value.dtype}, device={value.device}, "
			f"requires_grad={value.requires_grad})"
		)
		return

	if isinstance(value, Mapping):
		print(f"{prefix}{name}: {type(value).__name__} ({len(value)} entries)")
		for key, child in value.items():
			describe(child, repr(key), indent + 2)
		return

	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		print(f"{prefix}{name}: {type(value).__name__} ({len(value)} entries)")
		for index, child in enumerate(value):
			describe(child, f"[{index}]", indent + 2)
		return

	print(f"{prefix}{name}: {type(value).__name__} = {value!r}")


def tensor_items(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
	"""Return all tensors in a checkpoint with their mapping paths."""
	if isinstance(value, torch.Tensor):
		return [(prefix or "tensor", value)]

	if isinstance(value, Mapping):
		items: list[tuple[str, torch.Tensor]] = []
		for key, child in value.items():
			name = f"{prefix}.{key}" if prefix else str(key)
			items.extend(tensor_items(child, name))
		return items

	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		items = []
		for index, child in enumerate(value):
			name = f"{prefix}.{index}" if prefix else str(index)
			items.extend(tensor_items(child, name))
		return items

	return []


def c_identifier(name: str) -> str:
	"""Convert a checkpoint key into a valid C identifier."""
	identifier = re.sub(r"[^0-9A-Za-z_]", "_", name)
	if identifier and identifier[0].isdigit():
		identifier = f"weight_{identifier}"
	return identifier or "weight"


def format_values(values: list[int], values_per_line: int = 12) -> str:
	lines = []
	for start in range(0, len(values), values_per_line):
		chunk = values[start : start + values_per_line]
		lines.append("\t" + ", ".join(str(value) for value in chunk))
	return ",\n".join(lines)


def print_first_layer_raw(tensors: list[tuple[str, torch.Tensor]]) -> None:
	"""Print the unquantized first-layer weights and bias with three sig figs."""
	by_name = dict(tensors)
	for name in ("stem.weight", "stem.bias"):
		if name not in by_name:
			print(f"{name} was not found; skipping raw float32 output")
			continue

		tensor = by_name[name].detach().cpu().to(torch.float32)
		values = ", ".join(f"{float(value):.3g}f" for value in tensor.flatten())
		print(f"{name} raw float32 data, shape={tuple(tensor.shape)}:")
		print("[" + values + "]")


def write_header(tensors: list[tuple[str, torch.Tensor]], output: Path) -> None:
	"""Write symmetrically quantized tensors and their scales to a C header."""
	lines = [
		"#pragma once",
		"",
		"#include <stdint.h>",
		"",
		"// Quantization: float_value = int16_value * <name>_scale.",
		"",
	]

	for name, source in tensors:
		tensor = source.detach().cpu().to(torch.float32)
		max_abs = float(tensor.abs().max()) if tensor.numel() else 0.0
		scale = max_abs / 32767.0 if max_abs else 1.0
		quantized = torch.round(tensor / scale).clamp(-32767, 32767).to(torch.int16)
		identifier = c_identifier(name)
		values = quantized.flatten().tolist()

		lines.extend(
			[
				f"// {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
				f"static const float {identifier}_scale = {scale:.9g}f;",
				f"static const int16_t {identifier}[] = {{",
				format_values(values),
				"};",
				f"static const uint32_t {identifier}_count = {len(values)}u;",
				"",
			]
		)

	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"checkpoint",
		nargs="?",
		type=Path,
		default=DEFAULT_CHECKPOINT,
		help=f"Path to a PyTorch checkpoint (default: {DEFAULT_CHECKPOINT.name})",
	)
	parser.add_argument(
		"--output",
		"-o",
		type=Path,
		default=DEFAULT_HEADER,
		help=f"Output C header (default: {DEFAULT_HEADER})",
	)
	args = parser.parse_args()

	checkpoint_path = args.checkpoint.expanduser().resolve()
	if not checkpoint_path.is_file():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	print(f"Checkpoint: {checkpoint_path}")
	print("Loading on CPU with weights_only=True")
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
	tensors = tensor_items(checkpoint)
	if not tensors:
		raise ValueError("Checkpoint does not contain any tensors")

	print(f"Found {len(tensors)} tensors")
	print_first_layer_raw(tensors)
	write_header(tensors, args.output.expanduser().resolve())
	print(f"Wrote quantized weights to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
	main()
