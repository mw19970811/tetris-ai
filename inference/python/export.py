"""Export trained PyTorch model to ONNX for C++ inference."""

import torch
import argparse
from pathlib import Path


def export_to_onnx(checkpoint_path: str, output_path: str,
                   num_actions: int = 112, feature_dim: int = 53,
                   use_fp16: bool = False, opset_version: int = 14):
    """Export a trained DuelingDQN or ActorCritic model to ONNX format.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        output_path: Path for output .onnx file.
        num_actions: Number of action outputs.
        feature_dim: Feature vector dimension.
        use_fp16: Export in FP16 precision.
        opset_version: ONNX opset version.
    """
    from agent.model import DuelingDQN, ActorCritic

    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Detect model type from keys.
    state_dict = None
    if "agent_state" in checkpoint:
        state_dict = checkpoint["agent_state"]["online_net"]
        model = DuelingDQN(num_actions=num_actions, feature_dim=feature_dim, use_noisy=False)
        model.load_state_dict(state_dict)
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model = DuelingDQN(num_actions=num_actions, feature_dim=feature_dim, use_noisy=False)
        model.load_state_dict(state_dict)
    elif "online_net" in checkpoint:
        state_dict = checkpoint["online_net"]
        model = DuelingDQN(num_actions=num_actions, feature_dim=feature_dim, use_noisy=False)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

    model.eval()

    # Create dummy inputs.
    dummy_board = torch.randn(1, 1, 22, 10)
    dummy_features = torch.randn(1, feature_dim)

    # Dynamic axes for batching.
    dynamic_axes = {
        "board": {0: "batch"},
        "features": {0: "batch"},
        "q_values": {0: "batch"},
    }

    # Export (dynamo=False for broader PyTorch version compatibility).
    torch.onnx.export(
        model,
        (dummy_board, dummy_features),
        output_path,
        input_names=["board", "features"],
        output_names=["q_values"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"[Export] Model exported to {output_path}")

    # Verify.
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    board_in = dummy_board.numpy().astype("float32")
    feat_in = dummy_features.numpy().astype("float32")
    ort_out = session.run(None, {"board": board_in, "features": feat_in})
    torch_out = model(dummy_board, dummy_features).detach().numpy()
    diff = float(abs(ort_out[0] - torch_out).max())
    print(f"[Export] Verification max diff: {diff:.6f}")
    if diff < 1e-3:
        print("[Export] Verification PASSED.")
    else:
        print("[Export] Verification WARNING: difference > 1e-3")


def main():
    parser = argparse.ArgumentParser(description="Export Tetris AI model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--output", "-o", type=str, default="tetris_ai.onnx",
                        help="Output ONNX path")
    parser.add_argument("--num-actions", type=int, default=112)
    parser.add_argument("--feature-dim", type=int, default=53)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output,
                   num_actions=args.num_actions,
                   feature_dim=args.feature_dim,
                   opset_version=args.opset)


if __name__ == "__main__":
    main()
