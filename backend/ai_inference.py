#!/usr/bin/env python3

import json
import torch
import torch.nn.functional as F

# If you need more from these libs, import as necessary:
# from datasets import load_dataset
# import circuitsvis as cv
# import transformer_lens.utils as utils
# from tqdm import tqdm
# etc...

from transformer_lens import HookedTransformer
from sae_lens import SAE

# ------------------------------------
# 1) Load Model and SAEs
# ------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Gemma 9B model...")
model = HookedTransformer.from_pretrained("gemma-2-9b", device=DEVICE)
model.eval()
pad_token_id = model.tokenizer.pad_token_id
bos_token_id = model.tokenizer.bos_token_id

print(f"pad_token_id = {pad_token_id}, bos_token_id = {bos_token_id}")

# Example layers you said you’re using:
LAYER_INDICES = [7, 14, 21, 40]

# Example L0 values (from your code). Adjust as needed:
L0S = [92, 67, 129, 125]

# Load an SAE for each layer
saes = []
for i, layer_idx in enumerate(LAYER_INDICES):
    # Adjust "gemma-scope-9b-pt-res" release if needed
    # This returns (sae_instance, config)
    sae, _ = SAE.from_pretrained(
        release="gemma-scope-9b-pt-res",
        sae_id=f"layer_{layer_idx}/width_16k/average_l0_{L0S[i]}",
        device=DEVICE
    )
    saes.append(sae)

# Just ensure we don’t accumulate grads:
for param in model.parameters():
    param.requires_grad_(False)

for sae in saes:
    for param in sae.parameters():
        param.requires_grad_(False)

# Helpful: build a quick dict from layer index -> sae
sae_by_layer = { s.cfg.hook_layer: s for s in saes }

# ------------------------------------
# 2) Hooking Logic
# ------------------------------------
def build_sae_hook_fn(
    sae,
    token_mask: torch.Tensor,
    # userLatents is a dict: {tokenIndex -> { latentID -> value } }
    userLatents: dict,
    # keepSet is a set of latIDs we keep (others zeroed)
    keepSet: set
):
    """
    Creates a hook function for a given SAE module. The function:
      - Encodes hidden states -> features
      - Adds/overrides them with userLatents
      - Zeros out features not in keepSet
      - Decodes back to hidden states
    """

    def hook_fn(value: torch.Tensor, hook):
        # shape of value = [batch=1, seqLen, embedDim]
        # 1) Encode to SAE space
        with torch.no_grad():
            features = sae.encode(value)

        # 2) token_mask = a boolean mask that is False for special tokens (BOS, padding)
        #    We only apply transformations where token_mask == True.
        #    So we’ll do: features[i, j] replaced if token_mask[j] is True
        #    But for simplicity, we can do it for all tokens that are True.

        seqLen = features.shape[1]
        dim = features.shape[2]

        # 3) Overwrite from userLatents
        # userLatents is {tokenIndex -> { latentID -> floatValue }}
        for t_idx, latDict in userLatents.items():
            t_idx_int = int(t_idx)
            if t_idx_int < 0 or t_idx_int >= seqLen:
                continue
            if not token_mask[t_idx_int]:
                continue
            # latDict is { "123": 0.123, "999": 10.0, ... }
            for latID_str, val in latDict.items():
                latID = int(latID_str)
                if 0 <= latID < dim:
                    features[0, t_idx_int, latID] = val

        # 4) Zero out latents not in keepSet (only if token_mask[t_idx] is True)
        #    This means latents that are not in keepSet get set to 0
        #    But if keepSet is empty, everything is zero for that token
        for t in range(seqLen):
            if not token_mask[t]:
                continue
            # One approach:
            before = features[0, t]  # shape [dim]
            # Make a copy
            after = torch.zeros_like(before)
            # Only copy the kept latents:
            for latID in keepSet:
                if 0 <= latID < dim:
                    after[latID] = before[latID]
            # Overwrite
            features[0, t] = after

        # 5) Decode back to residual space
        with torch.no_grad():
            out = sae.decode(features)

        # 6) Put back into the hidden states only for tokens we are modifying
        for t in range(seqLen):
            if token_mask[t]:
                value[0, t] = out[0, t]

        return value

    return hook_fn

def build_hooks_list(
    token_mask: torch.Tensor,
    # requestedLatents: { layerIndex -> { tokenIndex -> { latentID -> float } } }
    requestedLatents: dict,
    # toggles: { layerIndex -> { tokenIndex -> [list_of_kept_latIDs] }}
    toggles: dict
):
    """Build a list of (hook_name, hook_fn) for each layer’s SAE."""
    fwd_hooks = []
    for layer_idx in LAYER_INDICES:
        # Hook name in HookedTransformer is typically: f"blocks.{layer_idx}.hook_resid_post"
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        if hook_name not in model.hook_dict:
            # Just skip if mismatch
            continue

        sae = sae_by_layer.get(layer_idx, None)
        if sae is None:
            continue

        # The user latents for this layer: { tokenIndex -> { latentID -> val } }
        layerLatents = requestedLatents.get(str(layer_idx), {})
        # The toggles for this layer: { tokenIndex -> [keptLatIDs] }
        layerToggles = toggles.get(str(layer_idx), {})

        # We want a single set of all latIDs across all tokens we keep
        # Because the reference code zeroes out latIDs not in toggles for each token
        # but we can do it per token if you want. Let’s do it per token for correctness:
        # => We’ll handle that inside the hook via a dictionary as well.

        # We'll actually produce a single hook function that sees the entire
        # (batch=1, seqLen, embedDim) and does the override. We pass in the dictionaries.
        # We need a separate keepSet per token, but to keep it simpler, let's do
        # a single keepSet across all tokens. (You can refine if you want “per token.”)
        # For closer parity with your original code, let's do full token iteration.

        def combined_hook_fn(value: torch.Tensor, hook):
            with torch.no_grad():
                # encode
                features = sae.encode(value)

            seq_len = features.shape[1]
            dim = features.shape[2]

            # Overwrite requested latents
            for t_idx_str, latDict in layerLatents.items():
                t_idx = int(t_idx_str)
                if t_idx < 0 or t_idx >= seq_len:
                    continue
                if not token_mask[t_idx]:
                    continue

                for latID_str, val in latDict.items():
                    latID = int(latID_str)
                    if 0 <= latID < dim:
                        features[0, t_idx, latID] = val

            # Zero out latents not toggled on
            for t_idx in range(seq_len):
                if not token_mask[t_idx]:
                    continue
                # toggles for this token
                keep_list = layerToggles.get(str(t_idx), [])
                keep_set = set(int(x) for x in keep_list)
                before = features[0, t_idx].clone()
                after = torch.zeros_like(before)
                for latID in keep_set:
                    if 0 <= latID < dim:
                        after[latID] = before[latID]
                features[0, t_idx] = after

            # decode
            with torch.no_grad():
                out = sae.decode(features)

            # set it back
            for t_idx in range(seq_len):
                if token_mask[t_idx]:
                    value[0, t_idx] = out[0, t_idx]

            return value

        # Add the hook
        # If you prefer separate hooking logic for each layer, do so:
        # We’ll add one hook per layer
        fwd_hooks.append((hook_name, combined_hook_fn))

    return fwd_hooks

# ------------------------------------
# 3) Utility: Collect final activations
# ------------------------------------
def gather_final_latent_activations(
    tokens: torch.Tensor,
    requestedLatents: dict
) -> list:
    """
    Construct a data structure similar to your "latent_activations".
    For each token t, for each layer in LAYER_INDICES, we return a dict of
    { latentID -> final numeric value } for only the latents the user requested.
    
    Because we've already injected them, we can run another pass *with*
    a "capture" hook to see final values. Alternatively, we can store them
    inside the main forward pass. For clarity, let's do a second pass:
    """
    # Build an empty structure: a list of length seqLen, each is
    #   [ {latentID: val for that layer}, { ... next layer ... }, ...]
    # But in your code, you only returned the latents that were requested, not all.
    # We'll do that for minimal overhead.

    # We'll define a single forward pass with partial hooks that read out
    # "blocks.{layer}.hook_resid_post" after the model runs. Then we’ll
    # store the latents for the tokens that were requested.

    # For convenience, let's merge all requested latIDs for each layer & token
    # so we know which to record.
    # requestedLatents: { layer_str -> { token_str -> [latentID -> float] } }
    # We can treat them as sets again:
    record_map = {}
    for layer_str, tokMap in requestedLatents.items():
        layer_idx = int(layer_str)
        for tok_str, latDict in tokMap.items():
            t_idx = int(tok_str)
            for latID_str in latDict.keys():
                latID = int(latID_str)
                # store in record_map[layer_idx][t_idx] -> set of latIDs
                if layer_idx not in record_map:
                    record_map[layer_idx] = {}
                if t_idx not in record_map[layer_idx]:
                    record_map[layer_idx][t_idx] = set()
                record_map[layer_idx][t_idx].add(latID)

    # We'll define a hook that, at each relevant layer, reads out the final activations
    captured = { lyr: None for lyr in LAYER_INDICES }

    def make_capture_fn(layer_idx):
        def capture_fn(value, hook):
            # store shape [batch=1, seqLen, embedDim]
            # clone it to CPU
            captured[layer_idx] = value.detach().cpu()
            return value
        return capture_fn

    # Build minimal list of hooks for capturing
    capture_hooks = []
    for layer_idx in LAYER_INDICES:
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        if hook_name in model.hook_dict:
            capture_hooks.append((hook_name, make_capture_fn(layer_idx)))

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=capture_hooks)

    # Now build the final data structure:
    # latent_activations[t][layerIndex] = { latentId: number }
    seqLen = tokens.shape[1]
    latent_activations = []
    for t in range(seqLen):
        # For each token t, we have a list of length len(LAYER_INDICES).
        # But note your original code had the same *order* of layers you used:
        layerList = []
        for layer_idx in LAYER_INDICES:
            # default empty dict
            latDict = {}
            if layer_idx in captured and captured[layer_idx] is not None:
                # shape [1, seqLen, embedDim]
                finalVals = captured[layer_idx][0, t]
                if layer_idx in record_map and t in record_map[layer_idx]:
                    for latID in record_map[layer_idx][t]:
                        latDict[str(latID)] = float(finalVals[latID].item())

            layerList.append(latDict)
        latent_activations.append(layerList)

    return latent_activations

# ------------------------------------
# 4) Main Inference Function
# ------------------------------------
def run_inference_with_latents(
    prompt: str,
    toggles: dict,
    requestedLatents: dict,
    top_k: int = 3
):
    """
    1) Tokenize the prompt
    2) Build a token_mask so we only alter normal tokens (skip BOS, etc.)
    3) Build forward hooks that override latents & zero out un‐toggled latents
    4) Run the forward pass, get final logits
    5) Gather final latent_activations
    6) Return topK probabilities, tokens, etc.
    """

    # 1) Tokenize
    with torch.no_grad():
        tokens = model.to_tokens(prompt, prepend_bos=True)
        # shape is [1, seqLen]. We assume batch=1

    # 2) Build token_mask
    # Example: mask all except BOS
    seqLen = tokens.shape[1]
    token_mask = torch.ones(seqLen, dtype=torch.bool)
    for i in range(seqLen):
        if tokens[0, i].item() == bos_token_id:
            token_mask[i] = False

    # 3) Build hooks
    hooks = build_hooks_list(token_mask, requestedLatents, toggles)

    # 4) Run forward pass
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
        # final logits = shape [batch=1, seqLen, vocab_size]
        final_logits = logits[0, -1, :]  # last position

    # 5) Gather final latents (only for the tokens/layers/IDs that were requested)
    latent_activations = gather_final_latent_activations(tokens, requestedLatents)

    # 6) Top k probabilities at the last token
    probs = F.softmax(final_logits, dim=-1)
    topk = torch.topk(probs, k=top_k)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()
    # Convert to strings
    top_tokens = [model.to_str_tokens(idx) for idx in top_indices]
    top_probs = dict()
    for i in range(top_k):
        # The token might be multiple sub-tokens, but typically top_tokens[i] is a single str
        # If it’s a single integer -> string conversion
        top_probs[top_tokens[i] if isinstance(top_tokens[i], str) else str(top_tokens[i])] = top_values[i]

    # Also gather the entire generated tokens as strings
    token_strs = model.to_str_tokens(tokens[0])

    return {
        "prompt": prompt,
        "tokens": token_strs,
        "toggles": toggles,
        "requestedLatents": requestedLatents,
        "result": {
            "topProbs": top_probs,
            "latentActivations": latent_activations
        }
    }

# ------------------------------------
# 5) Minimal CLI / Test
# ------------------------------------
if __name__ == "__main__":
    # Simple usage example:
    # python inference_sae.py
    example_prompt = '>>> prices = {"Apples": 3.5}\n>>> prices["Apples"]\n'
    example_toggles = {
        # Keep latents for layer 7, token #3 (?), we’ll say keep latIDs [42,100]
        "7": {
            "3": ["42", "100"]
        }
    }
    example_requestedLatents = {
        # Provide latents for layer 7, token #3, setting latID=42->0.777, latID=100->9.99
        "7": {
            "3": {
                "42": 0.777,
                "100": 9.99
            }
        }
    }

    out = run_inference_with_latents(example_prompt, example_toggles, example_requestedLatents, top_k=3)
    print(json.dumps(out, indent=2))
