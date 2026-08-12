"""Verify that the project's actual Transformer can train on Apple MPS.

Run ``python3 mps_smoke_test.py --require-mps`` on an Apple Silicon machine
before launching a long training run. The test compares CPU and MPS results,
then performs one optimizer step on each backend.
"""

import argparse

import torch

from char_transformer import CharacterTransformer
from device import mps_is_available


def run_one_step(device: torch.device, model_state: dict, config: dict, inputs: torch.Tensor, targets: torch.Tensor):
    model = CharacterTransformer(**config).to(device)
    model.load_state_dict(model_state)
    model.eval()  # Disable dropout so CPU and MPS calculations are comparable.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    logits, loss = model(inputs.to(device), targets.to(device))
    loss.backward()
    if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
        raise AssertionError(f"{device} produced a non-finite gradient")

    before = model.token_embedding.weight.detach().clone()
    optimizer.step()
    change = (model.token_embedding.weight.detach() - before).abs().max().item()
    if change == 0:
        raise AssertionError(f"{device} optimizer step did not change model weights")
    return logits.detach().cpu(), loss.detach().cpu(), change


def main():
    parser = argparse.ArgumentParser(description="Compare a small real Transformer step on CPU and Apple MPS.")
    parser.add_argument("--require-mps", action="store_true", help="Fail instead of skip when MPS is unavailable")
    parser.add_argument("--checkpoint", help="Optional checkpoint whose architecture and weights should be tested")
    args = parser.parse_args()

    if not mps_is_available():
        message = "MPS is unavailable in this PyTorch build or on this machine."
        if args.require_mps:
            raise RuntimeError(message)
        print(f"SKIPPED: {message}")
        return

    torch.manual_seed(7)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state" not in checkpoint or "model_config" not in checkpoint:
            raise ValueError("Checkpoint must contain model_state and model_config")
        config = checkpoint["model_config"]
        state = checkpoint["model_state"]
    else:
        config = {"vocab_size": 64, "block_size": 16, "embed_size": 32, "heads": 4, "layers": 2, "dropout": 0.0}
        state = CharacterTransformer(**config).state_dict()

    sequence_length = min(16, config["block_size"])
    inputs = torch.randint(0, config["vocab_size"], (2, sequence_length))
    targets = torch.randint(0, config["vocab_size"], (2, sequence_length))

    cpu_logits, cpu_loss, cpu_change = run_one_step(torch.device("cpu"), state, config, inputs, targets)
    mps_logits, mps_loss, mps_change = run_one_step(torch.device("mps"), state, config, inputs, targets)
    torch.testing.assert_close(cpu_logits, mps_logits, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(cpu_loss, mps_loss, rtol=1e-4, atol=1e-5)
    print(f"PASS: CPU loss {cpu_loss.item():.6f}; MPS loss {mps_loss.item():.6f}")
    print(f"Embedding updates: CPU {cpu_change:.6g}; MPS {mps_change:.6g}")


if __name__ == "__main__":
    main()
