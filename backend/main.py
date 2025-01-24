# main.py

import uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Union

# Import the relevant pieces from your inference.py
# (Adjust the import path as needed to match your project’s structure)
from inference import (
    tokenizer,               # The global HF tokenizer
    d_sae,                   # The latent dimension used by the SAEs
    simple_run,              # The high-level function that runs ablation + inference
)

################################################################################
# 1) Initialize app and set up CORS
################################################################################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, tighten restrictions here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################################################################################
# 2) /tokenize endpoint
################################################################################
@app.post("/tokenize")
def tokenize_endpoint(data: Dict[str, Union[str, List[str]]] = Body(...)):
    """
    Expects JSON with {"prompt": <string>}.
    Returns the raw tokens (as text) from the HF tokenizer.
    """
    prompt = data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="Invalid or missing `prompt`")

    # Tokenize the prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    # Convert IDs -> string tokens
    token_ids = encoded["input_ids"][0].tolist()
    tokens = [tokenizer._convert_id_to_token(tid) for tid in token_ids]

    return {
        "prompt": prompt,
        "tokens": tokens
    }

################################################################################
# 3) /runWithLatentMask endpoint
################################################################################
@app.post("/runWithLatentMask")
def run_with_latent_mask(data: Dict = Body(...)):
    """
    Expects JSON like:
      {
        "prompt": <str>,
        "toggles": {
          "<layer_idx>": {
            "<token_idx>": { "<latent_idx>": <mask_value>, ... },
            ...
          },
          ...
        },
        "requestedLatents": { ... },   # optional dictionary for 'returning' latents
        "zeroToggles": {               # optional dictionary specifying which latents
          "<layer_idx>": {
            "<token_idx>": { "<latent_idx>": <some_nonzero_value_to_zero>, ... },
            ...
          },
          ...
        }
      }

    We'll interpret any nonzero toggles[layer_idx][token_idx][latent_id] as "keep" (mask=1.0).
    All latents not in toggles => ablated to mean.

    Then, if zeroToggles is provided, any nonzero entry there is forcibly zeroed
    after the mean ablation step.
    """
    prompt = data.get("prompt")
    toggles = data.get("toggles", {})
    requested_latents = data.get("requestedLatents", {})
    zero_toggles = data.get("zeroToggles", {})

    # Validate required fields for the primary "keep" toggles
    if not prompt or not toggles:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: `prompt` or `toggles`"
        )

    # ----------------------------------------------------------------------------
    # (A) Convert toggles -> keep_dict (for mean ablation)
    # ----------------------------------------------------------------------------
    keep_dict = {}
    for layer_str, token_dict in toggles.items():
        layer_idx = int(layer_str)
        keep_dict[layer_idx] = {}

        for token_str, latent_map in token_dict.items():
            token_idx = int(token_str)
            latents_to_keep = []
            for latent_str, mask_val in latent_map.items():
                if mask_val != 0.0:  # Non-zero => keep this latent
                    latents_to_keep.append(int(latent_str))

            if len(latents_to_keep) > 0:
                keep_dict[layer_idx][token_idx] = latents_to_keep

    # ----------------------------------------------------------------------------
    # (B) Convert zeroToggles -> zero_dict (for zero ablation)
    # ----------------------------------------------------------------------------
    zero_dict = {}
    if zero_toggles:
        for layer_str, token_dict in zero_toggles.items():
            layer_idx = int(layer_str)
            zero_dict[layer_idx] = {}

            for token_str, latent_map in token_dict.items():
                token_idx = int(token_str)
                latents_to_zero = []
                for latent_str, val in latent_map.items():
                    if val != 0.0:  # Non-zero => forcibly zero out
                        latents_to_zero.append(int(latent_str))

                if len(latents_to_zero) > 0:
                    zero_dict[layer_idx][token_idx] = latents_to_zero
    else:
        zero_dict = None

    # ----------------------------------------------------------------------------
    # (C) Run the actual inference with ablation
    # ----------------------------------------------------------------------------
    top_tokens_dict, final_activations = simple_run(
        text=prompt,
        latents_dict=keep_dict,
        requested_return_dict=requested_latents,
        zero_latents_dict=zero_dict  # <--- new argument for zero ablations
    )

    # ----------------------------------------------------------------------------
    # (D) Format the top tokens result
    # ----------------------------------------------------------------------------
    # `top_tokens_dict` is something like { "<token_str>": probability_value, ... }
    top_probs = top_tokens_dict

    # ----------------------------------------------------------------------------
    # (E) Tokenize prompt for returning tokens in the response
    # ----------------------------------------------------------------------------
    encoded = tokenizer(prompt, return_tensors="pt")
    token_ids = encoded["input_ids"][0].tolist()
    tokens_list = [tokenizer._convert_id_to_token(tid) for tid in token_ids]

    # ----------------------------------------------------------------------------
    # (F) Return the final JSON structure
    # ----------------------------------------------------------------------------
    return {
        "prompt": prompt,
        "tokens": tokens_list,
        "toggles": toggles,
        "zeroToggles": zero_toggles,
        "result": {
            "topProbs": top_probs,
            "latentActivations": final_activations
        },
    }

################################################################################
# 4) Optional: Run via `python main.py`
################################################################################
if __name__ == "__main__":
    # By default, this will run on http://127.0.0.1:8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
