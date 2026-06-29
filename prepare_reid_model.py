#!/usr/bin/env python3
"""One-shot: export OSNet-x1.0 → ONNX → instruct user to run trtexec.

OSNet-x1.0 is the full-size model (vs x0.25 which is 4× smaller).
It provides significantly better re-identification accuracy in crowded scenes
at the cost of ~4× more inference time — still well within Jetson Orin budget.

Run once before using zed-color.py:
    python3 prepare_reid_model.py

Then run the printed trtexec command to build the TensorRT engine (~5 min on Orin).
"""
import os
import sys
from unittest.mock import MagicMock

# torchreid imports torchvision at the top level (for data transforms),
# but we only need the model architecture + pretrained weights — no transforms.
# Patching torchvision before the import avoids the missing-module error on
# Jetson where torchvision may not be installed alongside the NVIDIA PyTorch wheel.
for _mod in (
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "torchvision.utils",
    "torchvision.datasets",
    "torchvision.models",
):
    sys.modules.setdefault(_mod, MagicMock())

import torch
import torchreid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH   = os.path.join(SCRIPT_DIR, "osnet_x1_reid.onnx")
ENGINE_PATH = os.path.join(SCRIPT_DIR, "osnet_x1_reid.engine")

print("Loading OSNet-x1.0 from torchreid (weights auto-downloaded if needed)...")
model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    loss="softmax",
    pretrained=True,
)
model.eval()

print(f"Exporting to ONNX: {ONNX_PATH}")
dummy = torch.zeros(1, 3, 256, 128)
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["embedding"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
)
print(f"ONNX saved: {ONNX_PATH}")
print()
print("Now run this to build the TensorRT engine (one-time, ~5 min on Orin):")
print()
print(
    f"  /usr/src/tensorrt/bin/trtexec --onnx={ONNX_PATH} "
    f"--saveEngine={ENGINE_PATH} "
    "--fp16 --workspace=1024"
)
print()
print(f"When done, {ENGINE_PATH} must exist before running zed-color.py.")
