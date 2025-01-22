#!/usr/bin/env python3

import os
import torch
import nnsight
from nnsight import NNsight
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Import your local SAE code
from sae_lens import SAE  # must exist in your environment

################################################################
# 1) LOAD MODEL IN 8-BIT & WRAP WITH NNSIGHT
################################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

# Optional: 8-bit config
quant_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading Gemma 9B (8-bit) ...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model_8bit = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    quantization_config=quant_config,
    device_map="auto"  # accelerate auto placement
)
model_8bit.eval()

# Wrap with nnsight
wrapped_model = NNsight(model_8bit)

################################################################
# 2) LOAD SAEs FOR LAYERS = [7, 14, 21, 40]
################################################################

# Example layer indices & L0 combos you had
LAYER_INDICES = [7, 14, 21, 40]
L0S = [92, 67, 129, 125]

saes = []
for i, layer_idx in enumerate(LAYER_INDICES):
    # Load the pre-trained SAE from your release
    sae, _ = SAE.from_pretrained(
        release="gemma-scope-9b-pt-res",
        sae_id=f"layer_{layer_idx}/width_16k/average_l0_{L0S[i]}",
        device=DEVICE
    )
    # Freeze
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    # We'll store the final "mean_ablation" here after we compute it
    # shape [d_sae]
    sae.mean_ablation = torch.zeros(sae.cfg.d_sae, device=DEVICE)
    sae.num_obs = 0  # how many tokens used for averaging

    saes.append(sae)

# Make a quick map layer_idx -> sae
sae_by_layer = {s.cfg.hook_layer: s for s in saes}

################################################################
# 3) HELPER: NNSIGHT DECORATOR FOR TENSOR-ONLY
################################################################

def nt(func):
    """
    nnsight returns either a Tensor or a (Tensor, extra...) tuple.
    This decorator ensures we only modify the first element if it's a tuple.
    """
    def wrapper(x):
        if isinstance(x, tuple):
            t = x[0]
            assert isinstance(t, torch.Tensor), \
                "nnsight only at depth=1. Found deeper nesting?"
            t_out = func(t)
            return (t_out, *x[1:])
        elif isinstance(x, torch.Tensor):
            return func(x)
        else:
            return x
    return wrapper

################################################################
# 4) CAPTURE SAE FEATURES TO COMPUTE MEAN ABLATION
################################################################

@nt
def capture_sae_features(x, layer_idx):
    """
    For the given layer's residual output (x),
    1) We encode it via the SAE.
    2) Accumulate a running average across all tokens.
    3) Then decode back to keep model output unchanged.
    """
    sae = sae_by_layer[layer_idx]
    with torch.no_grad():
        features = sae.encode(x)  # shape [batch, seqLen, d_sae]

    # Accumulate into sae.mean_ablation
    batch_size, seq_len, d_sae = features.shape
    # Flatten over batch & seq
    for b in range(batch_size):
        for t in range(seq_len):
            sae.num_obs += 1
            # Update running mean
            old_mean = sae.mean_ablation
            new_val = features[b, t]
            sae.mean_ablation = old_mean + (new_val - old_mean) / sae.num_obs

    # decode back so we don't alter forward pass
    with torch.no_grad():
        out = sae.decode(features)
    return out  # same shape as x

def compute_mean_ablation(text_list, max_steps=100):
    """
    Runs up to 'max_steps' samples from text_list, capturing
    the SAE features at each layer & computing a running average.
    """
    sample_count = 0

    for i, text in enumerate(text_list):
        if i >= max_steps:
            break

        # Tokenize
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        # nnsight trace
        with wrapped_model.trace(input_ids):
            for layer_idx in LAYER_INDICES:
                # Let's get the layer's output
                # *IF* it's a GPT2-like model, try wrapped_model.model.transformer.h.XX
                # But your sample code had 'wrapped_model.model.layers[20].output' for some architectures
                # For many HF models, "layers" might be model_8bit.model.layers[XX]
                # We'll attempt a generic approach if "model.layers" is the final:
                # Adjust if your architecture differs.
                layer_output = wrapped_model.model.transformer.h[layer_idx].output
                # If that doesn't exist, try model_8bit.model.layers[XX].output
                # or your actual architecture. For demonstration:
                nnsight.apply(
                    lambda t: capture_sae_features(t, layer_idx),
                    layer_output
                )

        sample_count += 1
        if sample_count % 10 == 0:
            print(f"[Compute Means] Processed {sample_count} texts...")

    print("Done computing means. Final mean_ablation set in each SAE.")

################################################################
# 5) TOGGLING + MEAN ABLATION
################################################################

@nt
def apply_sae_toggles_and_mean_ablation(x, layer_idx, toggles_map, requested_latents_map):
    """
    x is the residual for 'blocks.{layer_idx}.hook_resid_post'.
    We'll:
      1) Encode -> features
      2) Overwrite latents from requested_latents
      3) Mean-ablate latents not in toggles
      4) Decode -> new residual
    'toggles_map' => {tokenIndex -> [list_of_latIDs_to_keep]}
    'requested_latents_map' => {tokenIndex -> {latID -> value}}
    
    Assumes batch=1 usage; adapt if you have bigger batch logic.
    """
    sae = sae_by_layer[layer_idx]
    with torch.no_grad():
        features = sae.encode(x)  # [1, seqLen, d_sae]

    _, seq_len, d_sae = features.shape

    # If you're strictly single-batch:
    for t_idx in range(seq_len):
        # Overwrite requested latents
        if str(t_idx) in requested_latents_map:
            lat_dict = requested_latents_map[str(t_idx)]
            for lat_str, val in lat_dict.items():
                lat_id = int(lat_str)
                if 0 <= lat_id < d_sae:
                    features[0, t_idx, lat_id] = val

        # Then "mean ablate" everything not in toggles
        keep_set = set()
        if str(t_idx) in toggles_map:
            keep_set = set(int(l) for l in toggles_map[str(t_idx)])

        orig_row = features[0, t_idx].clone()
        new_row = sae.mean_ablation.clone()
        for lat_id in keep_set:
            if 0 <= lat_id < d_sae:
                new_row[lat_id] = orig_row[lat_id]
        features[0, t_idx] = new_row

    with torch.no_grad():
        out = sae.decode(features)

    return out

def run_inference_with_nnsight(prompt, toggles_dict, latents_dict, max_new_tokens=32):
    """
    1) Tokenize
    2) With nnsight.trace, apply toggles/mean ablation on each relevant layer
    3) Generate text
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    # We'll do a single trace so we can patch each layer's output
    with wrapped_model.trace(input_ids):
        # For each layer in your toggles/latents
        for layer_str in toggles_dict.keys():
            layer_idx = int(layer_str)
            # E.g. layer_output might be h[layer_idx].output (depending on your arch)
            layer_output = wrapped_model.model.transformer.h[layer_idx].output

            # Build partial function capturing toggles
            toggles_map = toggles_dict[layer_str]  # e.g. { "3": ["42","100"], ...}
            latents_map = latents_dict.get(layer_str, {})
            # Apply toggles + mean ablation
            nnsight.apply(
                lambda t: apply_sae_toggles_and_mean_ablation(
                    t, layer_idx, toggles_map, latents_map
                ),
                layer_output
            )
    # After the trace, the model has run forward with your toggles
    gen_out = model_8bit.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(gen_out[0], skip_special_tokens=True)

################################################################
# 6) MAIN DEMO
################################################################

if __name__ == "__main__":

    # A) Example: compute the mean ablation for each SAE
    # We'll pick some text samples to accumulate means
    sample_texts = [
        ">>> ages = {'Bob': 15}\n>>> ages['Bob']\n",
        ">>> prices = {'Banana': 2.5}\n>>> prices['Banana']\n",
        "This is just a test for capturing average latents in gemma.",
        "Hello world! I'm testing nnsight hooking for mean ablation.",
    ]
    compute_mean_ablation(sample_texts, max_steps=4)

    # B) Example toggles & latents for LAYER=7
    toggles_example = {
        "7": {
            "3": ["42", "100"],  # keep latID=42, 100 at token idx=3
            "4": ["66", "999"]
        },
        # If you also want to do layers 14, 21, 40, define them here too
        # "14": {...}, etc.
    }
    requested_latents_example = {
        "7": {
            "3": { "42": 1.234, "100": 5.678 }
        }
        # more layers or tokens...
    }

    # C) Run inference with toggles
    prompt_text = ">>> ages = {'Alice': 12}\n>>> ages['Alice']\n"
    output_text = run_inference_with_nnsight(
        prompt_text,
        toggles_example,
        requested_latents_example,
        max_new_tokens=20
    )
    print("=== Output ===")
    print(output_text)
