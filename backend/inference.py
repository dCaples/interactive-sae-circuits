# %%
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


import torch

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
        self.all_tokenized = tokenizer(all_prompts, return_tensors="pt", padding=True)['input_ids']

        # ----------------------------------------
        # Tokenize correct prompts and error prompts separately
        # ----------------------------------------
        correct_tokenized = tokenizer(correct_prompts, return_tensors="pt", padding=True)['input_ids']
        error_tokenized   = tokenizer(error_prompts,   return_tensors="pt", padding=True)['input_ids']

        # Move to device
        # self.correct_tokenized = {k: v.to(device) for k, v in correct_tokenized.items()}
        # self.error_tokenized   = {k: v.to(device) for k, v in error_tokenized.items()}

        # ----------------------------------------
        # Final non-pad index (last valid token)
        # ----------------------------------------
        # correct_mask = self.correct_tokenized['attention_mask']
        # error_mask   = self.error_tokenized['attention_mask']

        # self.correct_last_nonpad = correct_mask.sum(dim=-1) - 1  # shape [batch_size]
        # self.error_last_nonpad   = error_mask.sum(dim=-1) - 1    # shape [batch_size]

        # ----------------------------------------
        # Label extraction
        # ----------------------------------------
        def single_token_id(response_str):
            # Convert response to a single token ID (or 2 tokens, we pick the second if possible)
            t = tokenizer(str(response_str), return_tensors="pt")["input_ids"].to(device)
            if t.shape[1] >= 2:
                return t[0, 1]
            else:
                return t[0, -1]

        # Get responses for correct / error
        correct_responses = [ex["response"] for ex in self.correct_batch]
        error_responses   = [ex["response"] for ex in self.error_batch]

        # Single-token labels for correct / error
        correct_labels = [single_token_id(r) for r in correct_responses]
        error_labels   = [single_token_id(r) for r in error_responses]

        self.correct_labels = torch.tensor(correct_labels, dtype=torch.long, device=device)
        self.error_labels   = torch.tensor(error_labels,   dtype=torch.long, device=device)

        # ----------------------------------------
        # All labels in the same order as all_tokenized
        # ----------------------------------------
        all_responses = correct_responses + error_responses
        all_labels    = [single_token_id(r) for r in all_responses]
        self.all_labels = torch.tensor(all_labels, dtype=torch.long, device=device)


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
        quantization_config=quant_config,
        device_map="auto"  # for accelerate or bitsandbytes
    )

    # Wrap with NNsight
    wrapped_model = NNsight(model_raw)

    # Example: 4 SAEs
    layers = [7, 14, 21, 40]
    l0s   = [92, 67, 129, 125]
    saes  = []
    for layer, l0_val in zip(layers, l0s):
        # Adapt your SAE loading code as needed:
        # e.g. SAE.from_pretrained(...)
        # For demonstration, we create a dummy SAE below. Replace with real.
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

# %%
tokenizer, wrapped_model, saes, sae_dict, component_dict = load_model_and_saes()

# %%
extended_name_pool = [
    "Bob", "Sam", "Lilly", "Rob", "Alice", "Charlie", "Sally", "Tom", "Jake", "Emily", 
    "Megan", "Chris", "Sophia", "James", "Oliver", "Isabella", "Mia", "Jackson", 
    "Emma", "Ava", "Lucas", "Benjamin", "Ethan", "Grace", "Olivia", "Liam", "Noah", "Diego"
]
full_dataset = generate_extended_dataset(extended_name_pool, num_samples=200)



# %%
def assert_tuple(x):
    assert isinstance(x, tuple), "must be tuple tensor"


# %%
# mean_items = ContrastiveDatasetBatch(full_dataset[:30], tokenizer)

# mean_ablation_dict = {}


# with wrapped_model.trace(mean_items.all_tokenized):
#     for key, value in component_dict.items():
#         component = value
#         sae = sae_dict[key]
#         output = component.output
#         nnsight_apply(assert_tuple, output)
        
#         mean_ablation_dict[key] = sae.encode(output[0]).mean(dim=0).save()


# %%
mean_ablation_dict = torch.load("./mean_ablate.pt")

# %%
test_items = ContrastiveDatasetBatch(full_dataset[-10:], tokenizer)

d_sae = saes[0].cfg.d_sae
seq_len = test_items.all_tokenized.shape[-1]

assert seq_len == 65, "sequence length is expected to be 65"

# %%
def run_ablated_model(tokenized, sae_mask_dict):

    def ablated_sae(input, tokens, sae, mean_ablation, mask):
        special_tokens_mask = torch.isin(tokens, torch.tensor(tokenizer.all_special_ids, dtype=torch.int64)) # true on special tokens (ie BOS)
        sae_acts = sae.encode(input)
        mean_input_diff = sae_acts - mean_ablation # add this to mean_ablation to get original value
        masked_sae_acts = mean_ablation + mean_input_diff * mask # where mask=1, let the input pass through else mean ablate
        masked_sae_acts[special_tokens_mask] *= 0 # zero out special tokens for clarity
        sae_out = sae.decode(masked_sae_acts)
        sae_out = sae_out.to(torch.float16)
        sae_out[special_tokens_mask] = input[special_tokens_mask] # replace with non-sae acts on special toks
        return sae_out, masked_sae_acts

    sae_acts = {}
    with wrapped_model.trace(tokenized):
        for k,component in component_dict.items():
            sae = sae_dict[k]
            mean_ablation = mean_ablation_dict[k]
            mask = sae_mask_dict[k]
            output = component.output
            nnsight_apply(assert_tuple, output)
            
            sae_input = output[0] # tensor inside tuple
            sae_output, masked_sae_acts = ablated_sae(
                input = sae_input, 
                tokens = tokenized,
                sae = sae,
                mean_ablation=mean_ablation,
                mask=mask
                )
            sae_acts[k] = masked_sae_acts.save()
            component.output = (sae_output,)
        
        output = wrapped_model.output.save()        

    model_out = torch.topk(torch.softmax(output.logits[:, -1, :], dim=-1), k=3)

    top_tokens = [tokenizer.convert_ids_to_tokens(model_out.indices[i]) for i in range(len(model_out.indices))]
    top_values = model_out.values
    return top_tokens, top_values, sae_acts

# %%
def run_ablated_model_demo():
    sae_mask_dict = {}
    for k, v in sae_dict.items():
        sae_mask_dict[k] = torch.rand(seq_len, d_sae)<0.9

    tokens = test_items.all_tokenized[0:1]

    return run_ablated_model(tokens, sae_mask_dict)
run_ablated_model_demo()

# %%
def keep_dict_to_mask_tensor(keep_dict: dict, seq_len: int, d_sae: int) -> dict:
    """
    Reconstruct the (seq_len x d_sae) mask tensors from a nested dict of
    {layer_idx: { token_idx: [latent_idx1, latent_idx2, ...], ... }, ...}.

    - keep_dict: the nested dict created by mask_tensor_to_keep_dict
    - seq_len:   the sequence length (number of tokens)
    - d_sae:     the latent dimension

    Returns:
      A dictionary {layer_idx -> (seq_len x d_sae) mask_tensor}.
    """
    sae_mask_dict = {}

    for layer_idx, layer_dict in keep_dict.items():
        # Initialize a zero tensor for the mask
        mask_tensor = torch.zeros(seq_len, d_sae, dtype=torch.float16)

        # For each token_idx in that layer
        for token_idx, latent_indices in layer_dict.items():
            # For each latent dimension to keep
            for latent_idx in latent_indices:
                mask_tensor[token_idx, latent_idx] = 1.0

        sae_mask_dict[layer_idx] = mask_tensor

    return sae_mask_dict

def mask_tensor_to_value_dict(
    sae_mask_dict: dict, 
    discard_value: float = 0.0
) -> dict:
    """
    Convert a dictionary of {layer_idx -> (seq_len x d_sae) mask tensors}
    into a nested dict specifying which token & latent dims do NOT match the
    discard_value, and what those values are.

    For each layer’s mask, we look for entries != discard_value.
    Then we store them as:
      {
        layer_idx: {
          token_idx: {
            latent_idx: <mask_value>,
            ...
          },
          ...
        },
        ...
      }
    Args:
        sae_mask_dict: Dict of {layer_idx -> mask_tensor}, each mask_tensor of shape [seq_len, d_sae].
        discard_value: Any float value that should be treated as "discard." Defaults to 0.0.

    Returns:
        A nested dictionary of the structure described above, containing
        all entries that are not equal to discard_value.
    """
    value_dict = {}

    for layer_idx, mask_tensor in sae_mask_dict.items():
        # Find all positions where the mask is not the discard_value
        keep_positions = (mask_tensor != discard_value).nonzero(as_tuple=False)

        if keep_positions.shape[0] == 0:
            # No entries to keep => store empty dict
            value_dict[layer_idx] = {}
            continue

        layer_dict = {}
        for token_idx, latent_idx in keep_positions:
            token_idx = token_idx.item()
            latent_idx = latent_idx.item()

            # Get the actual mask value at that position
            val = mask_tensor[token_idx, latent_idx].item()

            if token_idx not in layer_dict:
                layer_dict[token_idx] = {}
            layer_dict[token_idx][latent_idx] = val

        value_dict[layer_idx] = layer_dict

    return value_dict



def simple_run(
    text: str,
    latents_dict: dict,
    requested_return_dict: dict,
):
    """
    1) Tokenize `text`.
    2) Build a 'keep' mask dict => everything not mentioned is ablated (0).
    3) Use run_ablated_model(...) with that mask.
    4) Re-encode the final hidden states at each relevant layer to get the final latents
       for exactly the positions we 'kept'.
    5) Return top tokens, top probabilities, and final-latents dictionary.

    Requirements:
    - run_ablated_model(tokenized, sae_mask_dict) must be in scope.
      * That function expects: 1 => pass original, 0 => ablate to mean.
    """

    # # 1) Tokenize text (batch=1)
    tokenized = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    seq_len = tokenized.shape[1]
    assert seq_len == 65, "expected seq len 65 for circuit demo"

    # 2) Convert latents_dict => mask_tensors (1=keep, 0=ablate)
    #    i.e. everything not in latents_dict is ablated (0)
    sae_mask_dict = keep_dict_to_mask_tensor(
        keep_dict=latents_dict,
        seq_len=seq_len,
        d_sae=d_sae,
    )

    # print(sae_mask_dict)


    # 3) Run the ablated model
    top_tokens, top_values, sae_acts = run_ablated_model(tokenized, sae_mask_dict)

    # elementwise product of mask and sae_acts
    saved_activations = {}
    for k, v in sae_acts.items():
        saved_activations[k] =  sae_acts[k][0]
        sae_mask_dict[k] = sae_mask_dict[k].to(torch.bool)
        saved_activations[k][~sae_mask_dict[k]] = -1
    saved_activations = mask_tensor_to_value_dict(saved_activations, discard_value=-1)
    

    # squeeze the batch dim
    top_tokens = top_tokens[0]
    top_values = top_values[0]



    top_tokens_dict = {}
    for i, token in enumerate(top_tokens):
        top_tokens_dict[token] = top_values[i].item()

    return(top_tokens_dict, saved_activations)



def main():
    dict_circuit = {7: {62: [10768, 11635]}, 14: {62: [1724, 1788, 2576, 3805, 4811, 4834, 6868, 8269, 8746, 9066, 11766, 12929, 15603], 63: [8746]}, 21: {62: [534, 6740, 7015, 11455], 63: [712, 3076, 5066, 5880, 8255, 9551, 10824, 11416, 12314], 64: [52, 712, 1197, 1408, 4351, 6650, 7192, 8082, 8127, 9551, 10003, 12314, 12598, 13546, 14515]}, 40: {64: [215, 266, 637, 1073, 1322, 1435, 2295, 2493, 2534, 2664, 2881, 2930, 2964, 2996, 3056, 3685, 3960, 4501, 4603, 4689, 4769, 5862, 6619, 6742, 7622, 7792, 8416, 8778, 9230, 9309, 9447, 9682, 10069, 10155, 10316, 10628, 10936, 10993, 11066, 11103, 11403, 11579, 11706, 11839, 12037, 12258, 12735, 12988, 13095, 13113, 13479, 13967, 14423, 14504, 15628, 15705, 15851, 16039]}}
    top_tokens_dict, sae_acts = simple_run(test_items.correct_batch[0]['prompt'], dict_circuit, dict_circuit)
    print("top tokens:")
    print(top_tokens_dict)
    print("sae acts:")
    print(sae_acts)

if __name__ == "__main__":
    main()