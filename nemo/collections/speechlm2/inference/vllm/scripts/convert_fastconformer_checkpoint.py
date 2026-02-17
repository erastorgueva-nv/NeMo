# Copyright 2025 NVIDIA. All rights reserved.

"""Convert FastConformer encoder weights from S2S checkpoint to vLLM format."""

import os
import json
import argparse

from safetensors.torch import save_file, load_file


def load_s2s_config(checkpoint_path: str) -> dict:
    """Load S2S model config from checkpoint directory.

    Args:
        checkpoint_path: Path to S2S checkpoint directory

    Returns:
        Dictionary with S2S config, or empty dict if not found
    """
    # Try different possible config file names
    config_names = ["model_config.json", "config.json"]

    for config_name in config_names:
        config_path = os.path.join(checkpoint_path, config_name)
        if os.path.exists(config_path):
            print(f"Loading S2S config from: {config_path}")
            with open(config_path, "r") as f:
                return json.load(f)

    print("No S2S config file found, will infer from weights")
    return {}


def extract_encoder_config_from_s2s(s2s_config: dict) -> dict:
    """Extract FastConformer encoder config from S2S model config.

    Args:
        s2s_config: Full S2S model configuration

    Returns:
        Dictionary with encoder-specific configuration
    """
    config = {}

    try:
        # Navigate to perception config: model.stt.model.perception
        perception_cfg = s2s_config.get("model", {}).get("stt", {}).get("model", {}).get("perception", {})

        # Get encoder config
        encoder_cfg = perception_cfg.get("encoder", {})

        # Extract attention context size
        att_context_size = encoder_cfg.get("att_context_size", [70, 0])
        if isinstance(att_context_size, list) and len(att_context_size) >= 2:
            config["att_left_ctx"] = att_context_size[0]
            config["att_right_ctx"] = att_context_size[1]

        # Get conv context size
        conv_context_size = encoder_cfg.get("conv_context_size", "causal")
        config["conv_context_size"] = conv_context_size

        # Get norm type
        conv_norm_type = encoder_cfg.get("conv_norm_type", "layer_norm")
        config["norm_type"] = conv_norm_type

        # Get d_model from modality adapter
        modality_adapter_cfg = perception_cfg.get("modality_adapter", {})
        if "d_model" in modality_adapter_cfg:
            config["d_model"] = modality_adapter_cfg["d_model"]

        # Get sample rate from data config
        data_cfg = s2s_config.get("model", {}).get("stt", {}).get("data", {})
        if "source_sample_rate" in data_cfg:
            config["sample_rate"] = data_cfg["source_sample_rate"]

        # Get window_stride from preprocessor config
        preprocessor_cfg = perception_cfg.get("preprocessor", {})
        if "window_stride" in preprocessor_cfg:
            config["window_stride"] = preprocessor_cfg["window_stride"]

        print(f"Extracted encoder config from S2S config: {json.dumps(config, indent=2)}")

    except Exception as e:
        print(f"Warning: Error extracting encoder config from S2S config: {e}")

    return config


def convert(s2s_checkpoint_path: str, outdir: str):
    """Convert S2S checkpoint to vLLM FastConformer format.

    Args:
        s2s_checkpoint_path: Path to S2S checkpoint directory containing model.safetensors
        outdir: Path to output directory
    """
    os.makedirs(outdir, exist_ok=True)

    # Determine checkpoint directory
    if os.path.isdir(s2s_checkpoint_path):
        checkpoint_dir = s2s_checkpoint_path
        safetensors_path = os.path.join(s2s_checkpoint_path, "model.safetensors")
    else:
        checkpoint_dir = os.path.dirname(s2s_checkpoint_path)
        safetensors_path = s2s_checkpoint_path

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Could not find safetensors file at {safetensors_path}")

    # Load S2S config if available
    s2s_config = load_s2s_config(checkpoint_dir)
    encoder_config_from_s2s = extract_encoder_config_from_s2s(s2s_config)

    # Get sample rate and window stride from config
    sample_rate = encoder_config_from_s2s.get("sample_rate", 16000)
    window_stride = encoder_config_from_s2s.get("window_stride", 0.01)
    print(f"Using sample_rate: {sample_rate}, window_stride: {window_stride}")

    print(f"Loading S2S checkpoint from: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    print(f"Loaded {len(state_dict)} total parameters")

    # Extract encoder weights (keep encoder. prefix)
    perception_prefix = "stt_model.perception."
    encoder_prefix = "stt_model.perception.encoder."
    weights = {}

    for key, value in state_dict.items():
        if key.startswith(encoder_prefix):
            # Strip stt_model.perception. but keep encoder.
            new_key = key[len(perception_prefix) :]
            weights[new_key] = value

    print(f"Extracted {len(weights)} encoder parameters")

    # Extract projection/adapter weights
    proj_w_key = "stt_model.perception.proj.weight"
    proj_b_key = "stt_model.perception.proj.bias"

    if proj_w_key in state_dict and proj_b_key in state_dict:
        weights["adapter.weight"] = state_dict[proj_w_key]
        weights["adapter.bias"] = state_dict[proj_b_key]
        print(f"Added adapter projection weights")
        print(f"  adapter.weight shape: {weights['adapter.weight'].shape}")
        print(f"  adapter.bias shape: {weights['adapter.bias'].shape}")

    # Extract decoder weights if present
    decoder_prefix = "stt_model.perception.decoder."
    for key, value in state_dict.items():
        if key.startswith(decoder_prefix):
            new_key = key[len(decoder_prefix) :]
            weights[f"decoder.{new_key}"] = value

    # Calculate input dimension
    subsampling_factor = 8
    hop_length = int(sample_rate * window_stride)
    input_dim = subsampling_factor * hop_length

    # Infer adapted_dimension from adapter weight if present
    adapted_dimension = None
    if "adapter.weight" in weights:
        adapted_dimension = weights["adapter.weight"].shape[0]

    # Build config from S2S config with defaults
    cfg = encoder_config_from_s2s
    flat_config = {
        "architectures": ["FastConformerCTC"],
        "model_type": "fastconformer_ctc",
        "hidden_size": cfg.get("d_model", 1024),
        "d_model": cfg.get("d_model", 1024),
        "n_layers": cfg.get("n_layers", 24),
        "n_heads": cfg.get("n_heads", 8),
        "ff_mult": cfg.get("ff_mult", 4),
        "k_conv": cfg.get("k_conv", 9),
        "att_left_ctx": cfg.get("att_left_ctx", 70),
        "att_right_ctx": cfg.get("att_right_ctx", 0),
        "use_bias": cfg.get("use_bias", False),
        "norm_type": cfg.get("norm_type", "layer_norm"),
        "xscale": cfg.get("xscale", False),
        "vocab_size": cfg.get("vocab_size", 1024),
        "subsampling": {
            "hop_length": hop_length,
            "subsampling_factor": subsampling_factor,
        },
        "custom_input_specs": [{"name": "audio", "dim": input_dim}],
        "custom_outputs": ["acoustic_emb"],
        "num_output_tokens_per_step": 1,
    }

    if adapted_dimension is not None:
        flat_config["adapted_dimension"] = adapted_dimension

    # Save weights as SafeTensors
    output_safetensors = os.path.join(outdir, "model.safetensors")
    save_file(weights, output_safetensors)
    print(f"Saved weights to: {output_safetensors}")

    # Print weight keys for verification
    print(f"\nWeight keys ({len(weights)} total):")
    for k in sorted(weights.keys())[:20]:
        print(f"  {k}")
    if len(weights) > 20:
        print(f"  ... and {len(weights) - 20} more")

    # Save model index
    weight_map = {name: "model.safetensors" for name in weights.keys()}
    index = {
        "metadata": {
            "total_size": sum(w.numel() * w.element_size() for w in weights.values())
        },
        "weight_map": weight_map,
    }
    index_path = os.path.join(outdir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved model index to: {index_path}")

    # Save config
    config_path = os.path.join(outdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(flat_config, f, indent=2)
    print(f"Saved config to: {config_path}")

    print(f"\nConversion complete! Output directory: {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FastConformer from S2S checkpoint to vLLM format"
    )
    parser.add_argument(
        "--s2s-checkpoint",
        type=str,
        required=True,
        help="Path to S2S checkpoint directory (containing model.safetensors and model_config.json)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    args = parser.parse_args()

    convert(args.s2s_checkpoint, args.outdir)


if __name__ == "__main__":
    main()
