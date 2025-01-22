import random
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Union

app = FastAPI()

# --------------------------------------------------
# 1) Configure CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 2) Mock Tokenizer
# --------------------------------------------------
memorized = {
    'predict: ages["Bob"]': ["predict:", "ages", "[", '"Bob"', "]"],
    'predict: prices["Apples"]': ["predict:", "prices", "[", '"Apples"', "]"],
    'predict: dictionary["Hello"]': ["predict:", "dictionary", "[", '"Hello"', "]"],
}

def mock_tokenize(prompt: str) -> List[str]:
    if prompt in memorized:
        return memorized[prompt]
    return prompt.split()

# --------------------------------------------------
# 3) Random or zero generation for latents
# --------------------------------------------------
def random_value() -> float:
    """Return 0 or a random float in [0, 10) with 50/50 probability."""
    return 0 if random.random() < 0.5 else 10 * random.random()

# --------------------------------------------------
# 4) Layers
# --------------------------------------------------
LAYERS = [7, 21, 40]

# --------------------------------------------------
# 5) Routes
# --------------------------------------------------

@app.post("/tokenize")
def tokenize_endpoint(data: Dict[str, Union[str, List[str]]] = Body(...)):
    prompt = data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="Invalid or missing `prompt`")

    tokens = mock_tokenize(prompt)
    return {"prompt": prompt, "tokens": tokens}

@app.post("/runWithLatentMask")
def run_with_latent_mask(data: Dict = Body(...)):
    """
    Expects:
      data = {
        "prompt": <str>,
        "toggles": {  # a nested dict specifying layer_idx->token_idx->latent_idx->mask_value
          "7": {
            "0": { "0": 1.0, "2": 0.25 },
            "1": { "1": -1.5 },
            ...
          },
          "21": { ... },
          ...
        },
        "requestedLatents": { # same or different structure telling us which latents to create
          "7": {
            "0": [0, 1, 2],
            "1": [1],
            ...
          },
          "21": { ... },
          ...
        }
      }
    Returns the final latent activations after applying the toggles (mask).
    """
    prompt = data.get("prompt")
    toggles = data.get("toggles", {})
    requested_latents = data.get("requestedLatents", {})

    # 1) Validate required fields
    if not prompt or not toggles or not requested_latents:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: prompt, toggles, requestedLatents"
        )

    # 2) Tokenize prompt
    tokens = mock_tokenize(prompt)
    num_tokens = len(tokens)

    # 3) Build minimal structure: latentActivations[t][layerIndex] = { latentId: randomNumber }
    latent_activations = []
    for t in range(num_tokens):
        # For each token, build a list (in parallel with LAYERS)
        sae_list = []
        for layer_id in LAYERS:
            latents_for_this_cell = requested_latents.get(str(layer_id), {}).get(str(t), [])
            lat_map = {}
            for latent_id in latents_for_this_cell:
                # Convert to str to match the toggles dict keys
                latent_id_str = str(latent_id)
                lat_map[latent_id_str] = random_value()
            sae_list.append(lat_map)
        latent_activations.append(sae_list)

    # 4) Apply toggles using the nested-dict structure
    #    toggles[layer_id][token_idx] = { latent_idx: mask_value, ... }
    for t in range(num_tokens):
        for layer_idx, layer_id in enumerate(LAYERS):
            # toggles_for_layer = toggles.get(str(layer_id), {})
            toggles_for_token = toggles.get(str(layer_id), {}).get(str(t), {})
            lat_map = latent_activations[t][layer_idx]

            for latent_id_str in list(lat_map.keys()):
                # If toggles_for_token is missing or has a zero-value, discard
                mask_val = toggles_for_token.get(latent_id_str, 0.0)
                if mask_val == 0.0:
                    lat_map[latent_id_str] = 0.0
                else:
                    # Option A: Keep the original random activation as is
                    pass

                    # Option B (commented out): Multiply by mask_val
                    # lat_map[latent_id_str] = lat_map[latent_id_str] * mask_val

                    # Option C (commented out): Replace with the mask_val
                    # lat_map[latent_id_str] = mask_val

    # 5) Dummy "topProbs"
    top_probs = {
        "Traceback": random.random(),
        "1": random.random(),
        ">>>": random.random()
    }

    # 6) Return final
    return {
        "prompt": prompt,
        "tokens": tokens,
        "toggles": toggles,
        "result": {
            "topProbs": top_probs,
            "latentActivations": latent_activations,
        },
    }
