"""
Smoke tests for AASIST model — no dataset required.
Covers: instantiation, forward pass shapes, eval mode, batch consistency.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from models.AASIST import Model


def load_model(eval_mode=False):
    with open(ROOT / "config/AASIST.conf") as f:
        config = json.load(f)
    model = Model(config["model_config"])
    if eval_mode:
        model.eval()
    return model


NB_SAMP = 64600  # from config


def test_model_instantiation():
    model = load_model()
    assert isinstance(model, Model)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model has no parameters"
    print(f"  Total parameters: {total_params:,}")


def test_forward_pass_shapes():
    model = load_model(eval_mode=True)
    batch_size = 4
    x = torch.randn(batch_size, NB_SAMP)
    with torch.no_grad():
        last_hidden, output = model(x)
    assert last_hidden.shape == (batch_size, 160), (
        f"Expected last_hidden (4, 160), got {last_hidden.shape}")
    assert output.shape == (batch_size, 2), (
        f"Expected output (4, 2), got {output.shape}")


def test_output_is_finite():
    model = load_model(eval_mode=True)
    x = torch.randn(2, NB_SAMP)
    with torch.no_grad():
        _, output = model(x)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"


def test_single_sample():
    model = load_model(eval_mode=True)
    x = torch.randn(1, NB_SAMP)
    with torch.no_grad():
        last_hidden, output = model(x)
    assert output.shape == (1, 2)


def test_train_forward_no_crash():
    model = load_model(eval_mode=False)
    x = torch.randn(2, NB_SAMP)
    last_hidden, output = model(x)
    loss = output.sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradient contains NaN/Inf"


def test_freq_aug_forward():
    model = load_model(eval_mode=False)
    x = torch.randn(2, NB_SAMP)
    last_hidden, output = model(x, Freq_aug=True)
    assert output.shape == (2, 2)


def test_output_classes():
    """Output dim 0=spoof, 1=bonafide — exactly 2 classes for binary detection."""
    model = load_model(eval_mode=True)
    x = torch.randn(1, NB_SAMP)
    with torch.no_grad():
        _, output = model(x)
    assert output.shape[-1] == 2, "Expected binary (spoof/bonafide) output"


def test_deterministic_in_eval():
    model = load_model(eval_mode=True)
    x = torch.randn(2, NB_SAMP)
    with torch.no_grad():
        _, out1 = model(x)
        _, out2 = model(x)
    assert torch.allclose(out1, out2), "Eval mode output is not deterministic"


if __name__ == "__main__":
    tests = [
        test_model_instantiation,
        test_forward_pass_shapes,
        test_output_is_finite,
        test_single_sample,
        test_train_forward_no_crash,
        test_freq_aug_forward,
        test_output_classes,
        test_deterministic_in_eval,
    ]
    passed = failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
