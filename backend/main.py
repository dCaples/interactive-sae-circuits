import random
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Union

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # You can restrict this to specific origins, e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 1) Mock Tokenize
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
# 2) Random/zero generation for latents
# --------------------------------------------------
def random_value() -> float:
    return 0 if random.random() < 0.5 else 10 * random.random()

# --------------------------------------------------
# 3) Layers
# --------------------------------------------------
LAYERS = [7, 21, 40]

# --------------------------------------------------
# 4) Routes
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

    # 3) Build minimal structure: latentActivations[t][layerIndex] = { latentId: number }
    latent_activations = []
    for t in range(num_tokens):
        sae_list = []
        for layer_id in LAYERS:
            latents_for_this_cell = requested_latents.get(str(layer_id), {}).get(str(t), [])
            lat_obj = {}
            for latent_id in latents_for_this_cell:
                lat_obj[latent_id] = random_value()
            sae_list.append(lat_obj)
        latent_activations.append(sae_list)

    # 4) Apply toggles
    for t in range(num_tokens):
        for idx, layer_id in enumerate(LAYERS):
            latents_we_keep = toggles.get(str(layer_id), {}).get(str(t), [])
            keep_set = set(latents_we_keep)

            lat_map = latent_activations[t][idx]
            for latent_id_str in list(lat_map.keys()):
                if latent_id_str not in keep_set:
                    lat_map[latent_id_str] = 0

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
