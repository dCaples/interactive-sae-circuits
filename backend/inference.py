import random
import torch
import torch.nn.functional as F
import json
from typing import Dict, List
from functools import lru_cache

# nnsight / transformers imports
import nnsight
from nnsight import NNsight
from nnsight import apply as nnsight_apply
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
torch.set_default_device('cuda')

# custom SAE code
from sae_lens import SAE


################################################################################
# 1. DATASET
################################################################################

def generate_extended_dataset(name_pool, num_samples=5):
    """
    Create a list of examples. Each example has a 'correct' and an 'error' dict lookup.
    """
    dataset = []
    for _ in range(num_samples):
        selected_names = random.sample(name_pool, 5)
        age_dict = {name: random.randint(10, 19) for name in selected_names}
        # correct
        correct_name = random.choice(list(age_dict.keys()))
        correct_prompt = (
            'Type "help", "copyright", "credits" or "license" for more information.\n'
            f">>> age = {age_dict}\n>>> age[\"{correct_name}\"]\n"
        )
        correct_response = age_dict[correct_name]
        # incorrect
        incorrect_name = random.choice([n for n in name_pool if n not in age_dict])
        if random.random() > 0.5:
            incorrect_prompt = (
                'Type "help", "copyright", "credits" or "license" for more information.\n'
                f">>> age = {age_dict}\n>>> age[\"{incorrect_name}\"]\n"
            )
        else:
            keys = list(age_dict.keys())
            items = list(age_dict.values())
            location = keys.index(correct_name)
            keys[location] = incorrect_name
            broken_age_dict = dict(zip(keys, items))
            incorrect_prompt = (
                'Type "help", "copyright", "credits" or "license" for more information.\n'
                f">>> age = {broken_age_dict}\n>>> age[\"{correct_name}\"]\n"
            )
        dataset.append({
            "correct": {
                "prompt": correct_prompt,
                "response": correct_response
            },
            "error": {
                "prompt": incorrect_prompt,
                "response": "Traceback"
            }
        })
    return dataset

class ContrastiveDatasetBatch:
    """
    - Takes a subset of items
    - Tokenizes
    - Stores last-token labels
    """
    def __init__(self, dataset_items, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        
        # Separate correct and error examples
        self.correct_batch = [item["correct"] for item in dataset_items]
        self.error_batch   = [item["error"]   for item in dataset_items]
        self.batch_size    = len(self.correct_batch)

        # Extract prompts
        correct_prompts = [ex["prompt"] for ex in self.correct_batch]
        error_prompts   = [ex["prompt"] for ex in self.error_batch]

        # ----------------------------------------
        # Tokenize all prompts together
        # ----------------------------------------
        all_prompts = correct_prompts + error_prompts
        self.all_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)['input_ids'].to(self.device)

        # (Omitted some code that used to be here - not strictly needed for the example)

        # ----------------------------------------
        # Single-token labels for correct / error
        # ----------------------------------------
        def single_token_id(response_str):
            # Convert response to a single token ID (or 2 tokens, we pick the second if possible)
            t = tokenizer(str(response_str), return_tensors="pt")["input_ids"].to(device)
            if t.shape[1] >= 2:
                return t[0, 1]
            else:
                return t[0, -1]

        correct_responses = [ex["response"] for ex in self.correct_batch]
        error_responses   = [ex["response"] for ex in self.error_batch]

        correct_labels = [single_token_id(r) for r in correct_responses]
        error_labels   = [single_token_id(r) for r in error_responses]
        
        self.correct_labels = torch.tensor(correct_labels, dtype=torch.long, device=self.device)
        self.error_labels   = torch.tensor(error_labels,   dtype=torch.long, device=self.device)

        all_responses = correct_responses + error_responses
        all_labels    = [single_token_id(r) for r in all_responses]
        self.all_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)


################################################################################
# 2. LOAD MODEL + SAES, AND WRAP WITH NNSIGHT
################################################################################

def load_model_and_saes():
    """
    - Load a quantized model with bitsandbytes
    - Wrap with NNsight
    - Create SAEs (4 layers)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    model_raw = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        # quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Wrap with NNsight
    wrapped_model = NNsight(model_raw)

    # Example: 4 SAEs
    layers = [7, 14, 21, 40]
    l0s   = [92, 67, 129, 125]
    saes  = []
    for layer, l0_val in zip(layers, l0s):
        sae_obj = SAE.from_pretrained(
            release="gemma-scope-9b-pt-res",
            sae_id=f"layer_{layer}/width_16k/average_l0_{l0_val}",
            device=device
        )[0]
        # freeze
        for p in sae_obj.parameters():
            p.requires_grad_(False)
        saes.append(sae_obj)

    # Freeze the main model’s parameters
    for param in model_raw.parameters():
        param.requires_grad_(False)
    
    sae_dict = {}
    component_dict = {}
    for i, layer in enumerate(layers):
        component_dict[layer] = wrapped_model.model.layers[layer]
        sae_dict[layer] = saes[i]

    return tokenizer, wrapped_model, saes, sae_dict, component_dict

tokenizer, wrapped_model, saes, sae_dict, component_dict = load_model_and_saes()

extended_name_pool = [
    "Bob", "Sam", "Lilly", "Rob", "Alice", "Charlie", "Sally", "Tom", "Jake", "Emily", 
    "Megan", "Chris", "Sophia", "James", "Oliver", "Isabella", "Mia", "Jackson", 
    "Emma", "Ava", "Lucas", "Benjamin", "Ethan", "Grace", "Olivia", "Liam", "Noah", "Diego"
]
full_dataset = generate_extended_dataset(extended_name_pool, num_samples=200)

test_items = ContrastiveDatasetBatch(full_dataset[-10:], tokenizer)
seq_len = test_items.all_tokenized.shape[-1]

# Suppose we already have the mean ablations stored
mean_ablation_dict = torch.load("./mean_ablate.pt")

d_sae = saes[0].cfg.d_sae
assert seq_len == 65, "sequence length is expected to be 65"


################################################################################
# 3. Ablation Helpers
################################################################################

def assert_tuple(x):
    assert isinstance(x, tuple), "must be tuple tensor"

def keep_dict_to_mask_tensor(keep_dict: dict, seq_len: int, d_sae: int) -> dict:
    """
    Reconstruct the (seq_len x d_sae) mask tensors from a nested dict of
    {layer_idx: { token_idx: [latent_idx1, latent_idx2, ...], ... }, ...}.
    Returns a {layer_idx -> (seq_len x d_sae) tensor} with 1.0=keep, 0.0=not-keep.
    """
    sae_mask_dict = {}
    for layer_idx, layer_dict in keep_dict.items():
        mask_tensor = torch.zeros(seq_len, d_sae, dtype=torch.float16)
        for token_idx, latent_indices in layer_dict.items():
            for latent_idx in latent_indices:
                mask_tensor[token_idx, latent_idx] = 1.0
        sae_mask_dict[layer_idx] = mask_tensor
    return sae_mask_dict

def zero_dict_to_mask_tensor(zero_dict: dict, seq_len: int, d_sae: int) -> dict:
    """
    Similar structure to `keep_dict_to_mask_tensor`, but for latents you want to set to zero.
    Returns a {layer_idx -> (seq_len x d_sae) tensor} with 1.0=zero out, 0.0=don’t zero out.
    """
    sae_zero_mask_dict = {}
    for layer_idx, layer_dict in zero_dict.items():
        mask_tensor = torch.zeros(seq_len, d_sae, dtype=torch.float16)
        for token_idx, latent_indices in layer_dict.items():
            for latent_idx in latent_indices:
                mask_tensor[token_idx, latent_idx] = 1.0
        sae_zero_mask_dict[layer_idx] = mask_tensor
    return sae_zero_mask_dict

def mask_tensor_to_value_dict(
    sae_mask_dict: dict, 
    discard_value: float = 0.0
) -> dict:
    """
    Convert a dictionary of {layer_idx -> (seq_len x d_sae) mask tensors}
    into a nested dict specifying which token & latent dims do NOT match the
    discard_value, and what those values are.
    """
    value_dict = {}
    for layer_idx, mask_tensor in sae_mask_dict.items():
        keep_positions = (mask_tensor != discard_value).nonzero(as_tuple=False)
        if keep_positions.shape[0] == 0:
            value_dict[layer_idx] = {}
            continue
        layer_dict = {}
        for token_idx, latent_idx in keep_positions:
            token_idx = token_idx.item()
            latent_idx = latent_idx.item()
            val = mask_tensor[token_idx, latent_idx].item()
            if token_idx not in layer_dict:
                layer_dict[token_idx] = {}
            layer_dict[token_idx][latent_idx] = val
        value_dict[layer_idx] = layer_dict
    return value_dict


################################################################################
# 4. run_ablated_model with optional zero ablation
################################################################################

def run_ablated_model(tokenized, sae_keep_mask_dict, sae_zero_mask_dict=None):
    """
    1) For each layer:
       - encode -> SAE latents
       - mean ablate everything that is not keep_mask=1
       - if sae_zero_mask_dict is provided, set those latents to 0 afterwards
       - decode -> layer hidden states
       - store the final SAE latents to sae_acts
    2) Return top tokens, probabilities, and final sae_acts
    """

    def ablated_sae(input_hidden, tokens, sae, mean_ablation, keep_mask, zero_mask):
        # encode to SAE space
        sae_acts = sae.encode(input_hidden)

        # difference from mean
        mean_input_diff = sae_acts - mean_ablation

        # mean ablation for not-kept latents
        masked_sae_acts = mean_ablation + mean_input_diff * keep_mask

        # optional step: zero out latents in zero_mask
        if zero_mask is not None:
            masked_sae_acts[:, zero_mask.bool()] = 0.0

        # decode back
        sae_out = sae.decode(masked_sae_acts).to(torch.float16)

        # Special tokens pass-through (if you want them unaffected)
        special_tokens_mask = torch.isin(
            tokens, 
            torch.tensor(tokenizer.all_special_ids, dtype=torch.int64, device=tokens.device)
        )
        sae_out[special_tokens_mask] = input_hidden[special_tokens_mask]

        return sae_out, masked_sae_acts

    sae_acts = {}

    with wrapped_model.trace(tokenized):
        for layer_idx, component in component_dict.items():
            sae = sae_dict[layer_idx]
            output = component.output
            nnsight_apply(assert_tuple, output)

            # retrieve the mean ablation vector for this layer
            mean_ablation = mean_ablation_dict[layer_idx]

            # get the keep mask for this layer
            if layer_idx in sae_keep_mask_dict:
                keep_mask = sae_keep_mask_dict[layer_idx]
            else:
                # if not specified, default to 0.0 => mean ablate everything
                keep_mask = torch.zeros_like(mean_ablation)

            # get zero mask for this layer (if any)
            zero_mask = None
            if sae_zero_mask_dict is not None:
                zero_mask = sae_zero_mask_dict.get(layer_idx, None)

            input_hidden = output[0]
            sae_out, masked_sae_acts = ablated_sae(
                input_hidden=input_hidden,
                tokens=tokenized,
                sae=sae,
                mean_ablation=mean_ablation,
                keep_mask=keep_mask,
                zero_mask=zero_mask
            )

            # store the final SAE latents (after zero-ablation if any)
            sae_acts[layer_idx] = masked_sae_acts.save()

            # override the output hidden states
            component.output = (sae_out,)

        # final forward pass from last layer
        final = wrapped_model.output.save()

    # get top tokens
    logits = final.logits
    model_out = torch.topk(torch.softmax(logits[:, -1, :], dim=-1), k=3)
    top_tokens = [tokenizer.convert_ids_to_tokens(model_out.indices[i]) for i in range(len(model_out.indices))]
    top_values = model_out.values

    return top_tokens, top_values, sae_acts


################################################################################
# 5. Simple Run
################################################################################

def simple_run(
    text: str,
    latents_dict: dict,
    requested_return_dict: dict,
    zero_latents_dict: dict | None = None
):
    """
    1) Tokenize `text`.
    2) Build a 'keep' mask dict => everything not mentioned is ablated to mean.
    3) Optionally build a 'zero' mask dict => latents here are forced to zero.
    4) Run the ablated model.
    5) Return top tokens, top probabilities, and final-latents dictionary.
    """

    # 1) Tokenize
    tokenized = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    seq_len = tokenized.shape[1]
    if seq_len != 65:
        print(f"Warning: expected seq_len=65 but got {seq_len}")

    # 2) Convert latents_dict => keep mask (1=keep, 0=ablate-to-mean)
    sae_keep_mask_dict = keep_dict_to_mask_tensor(
        keep_dict=latents_dict,
        seq_len=seq_len,
        d_sae=d_sae,
    )

    # 3) Convert zero_latents_dict => zero mask (1=force-to-zero, 0=do-nothing)
    if zero_latents_dict is not None:
        sae_zero_mask_dict = zero_dict_to_mask_tensor(
            zero_dict=zero_latents_dict,
            seq_len=seq_len,
            d_sae=d_sae
        )
    else:
        sae_zero_mask_dict = None

    # 4) Run the ablated model
    top_tokens, top_values, sae_acts = run_ablated_model(
        tokenized, 
        sae_keep_mask_dict=sae_keep_mask_dict, 
        sae_zero_mask_dict=sae_zero_mask_dict
    )

    # Post-process: Keep only the latents that we actually "kept" for output
    saved_activations = {}
    for layer_idx, final_latents in sae_acts.items():
        # final_latents is shape [seq_len, d_sae], for batch=1
        final_latents = final_latents[0]  # remove batch dim

        # convert keep_mask -> bool
        keep_mask = sae_keep_mask_dict.get(layer_idx, torch.zeros_like(final_latents)).bool()
        # set latents we do not keep to -1 so user can see they were not kept
        final_latents[~keep_mask] = -1

        # also if you want to reflect zero latents, you can do so,
        # but you'd see them directly in the final_latents anyway.

        saved_activations[layer_idx] = final_latents

    # Convert to nested dictionary structure
    saved_activations = mask_tensor_to_value_dict(saved_activations, discard_value=-1)

    # 5) Format the top tokens + values
    top_tokens = top_tokens[0]  # batch=1
    top_values = top_values[0]

    top_tokens_dict = {}
    for i, token in enumerate(top_tokens):
        top_tokens_dict[token] = top_values[i].item()

    return top_tokens_dict, saved_activations


################################################################################
# 6. Usage Example
################################################################################

def main():
    # Example "keep" dict (the circuit you want to keep)
    dict_circuit = {
        7:  {62: [10768, 11635]}, 
        14: {62: [1724, 1788], 63: [8746]}, 
        21: {62: [534, 6740], 63: [712, 3076], 64: [52, 712]},
        40: {64: [215, 266, 637]} 
    }

    # Example "zero" dict: latents to forcibly set to zero after mean ablation
    dict_zero = {
        14: {63: [8746]},  # forcibly set layer=14, token=63, latents=[8746] to zero
        40: {64: [637]}    # forcibly set layer=40, token=64, latents=[637] to zero
    }

    # Use the correct prompt from the last test item
    text = test_items.correct_batch[0]['prompt']

    top_tokens_dict, sae_acts = simple_run(
        text=text,
        latents_dict=dict_circuit,
        requested_return_dict=dict_circuit,
        zero_latents_dict=dict_zero
    )
    
    print("=== Top Tokens ===")
    print(top_tokens_dict)
    print("=== Final SAE Latents (Mask Dictionary) ===")
    print(sae_acts)


if __name__ == "__main__":
    main()