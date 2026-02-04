"""Export a HuggingFace model to ONNX for use with ASPIRE.

This script exports causal language models to ONNX format optimized
for inference on RTX 5080 (or other CUDA/DirectML devices).

Usage:
    # Export Phi-3-mini (recommended for testing)
    python scripts/export_model.py --model microsoft/Phi-3-mini-4k-instruct

    # Export with custom output path
    python scripts/export_model.py --model microsoft/Phi-3-mini-4k-instruct --output models/phi3.onnx

    # Export smaller model for faster iteration
    python scripts/export_model.py --model Qwen/Qwen2-0.5B-Instruct

Recommended models for ASPIRE (sorted by size):
    - Qwen/Qwen2-0.5B-Instruct     (~1GB)  - Fast, good for testing loop
    - microsoft/Phi-3-mini-4k-instruct (~2.5GB) - Good quality/speed balance
    - Qwen/Qwen2-1.5B-Instruct     (~3GB)  - Better quality
    - microsoft/Phi-3-small-8k-instruct (~4GB) - High quality

Note: Export can take several minutes depending on model size.
"""

import argparse
from pathlib import Path
import sys


def export_with_optimum(model_name: str, output_path: Path):
    """Export using Optimum library (recommended)."""
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError:
        print("❌ Optimum not installed. Install with:")
        print("   pip install 'optimum[onnxruntime-gpu]'")
        sys.exit(1)

    print(f"Exporting {model_name} with Optimum...")
    print("This may take several minutes...")

    # Export
    model = ORTModelForCausalLM.from_pretrained(
        model_name,
        export=True,
        trust_remote_code=True,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path.parent)

    # Find the actual .onnx file
    onnx_files = list(output_path.parent.glob("*.onnx"))
    if onnx_files:
        # Rename to requested name
        onnx_files[0].rename(output_path)
        print(f"✓ Exported to: {output_path}")
    else:
        print(f"✓ Exported to: {output_path.parent}")


def export_with_torch(model_name: str, output_path: Path):
    """Export using torch.onnx (fallback)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ Required packages not installed. Install with:")
        print("   pip install torch transformers")
        sys.exit(1)

    print(f"Exporting {model_name} with torch.onnx...")
    print("This may take several minutes...")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    # Dummy input
    dummy_input = tokenizer("Hello, world!", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"✓ Exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: models/<model-name>.onnx)",
    )
    parser.add_argument(
        "--method",
        choices=["optimum", "torch"],
        default="optimum",
        help="Export method (default: optimum)",
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_slug = args.model.replace("/", "-").lower()
        output_path = Path("models") / f"{model_slug}.onnx"

    print("=" * 60)
    print("ASPIRE Model Export")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Output: {output_path}")
    print(f"Method: {args.method}")
    print()

    if args.method == "optimum":
        export_with_optimum(args.model, output_path)
    else:
        export_with_torch(args.model, output_path)

    print()
    print("Next steps:")
    print(f"  python examples/demo_onnx_loop.py --model {output_path} --tokenizer {args.model}")


if __name__ == "__main__":
    main()
