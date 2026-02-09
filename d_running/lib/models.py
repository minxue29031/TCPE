import os
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

sys.path.append("./")
from huggingface.transcoder_adapter import TranscoderAdapter


def load_model(path, device_map=None):
    if device_map is None:
        device_map = "cuda"
    return AutoModelForCausalLM.from_pretrained("path_to_model", torch_dtype=torch.float16, device_map=device_map)


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained("path_to_model")


def load_model_and_tokenizer(path, device_map=None):
    model = load_model(path, device_map=device_map)
    tok = load_tokenizer(path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"<pad>"})
        model.config.pad_token_id = tok.pad_token_id
        model.model.padding_idx = model.config.pad_token_id
        model.generation_config.pad_token_id = tok.pad_token_id
        # potentially resize embedding and set padding idx
        new_embedding_size = max(len(tok.vocab), model.config.vocab_size)
        model.resize_token_embeddings(new_embedding_size)
        new_embedding = torch.nn.Embedding(new_embedding_size, model.config.hidden_size, tok.pad_token_id)
        old_embedding = model.get_input_embeddings()
        new_embedding.to(old_embedding.weight.device, old_embedding.weight.dtype)
        new_embedding.weight.data[:model.config.vocab_size] = old_embedding.weight.data
        model.set_input_embeddings(new_embedding)
        model.config.vocab_size = new_embedding_size
        # artifact needs to be kept so that it is detectable we modified the embeddings here
        # model.vocab_size = new_embedding_size
    
    return model, tok


def load_model_and_tokenizer_with_TC_layer(
    path, tc_layer_path, layer=19, device_map=None
):
    model = load_model("path_to_model", device_map=device_map)
    adapter = TranscoderAdapter.load(tc_layer_path)
    device = next(model.base_model.layers[layer].mlp.parameters()).device
    model.base_model.layers[layer].mlp = adapter.to(model.dtype).to(device)
    tok = load_tokenizer("path_to_model")
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
        model.config.pad_token_id = tok.pad_token_id
        model.model.padding_idx = model.config.pad_token_id
        model.generation_config.pad_token_id = tok.pad_token_id
 
        new_embedding_size = max(len(tok.vocab), model.config.vocab_size)
        model.resize_token_embeddings(new_embedding_size)
        new_embedding = torch.nn.Embedding(new_embedding_size, model.config.hidden_size, tok.pad_token_id)
        old_embedding = model.get_input_embeddings()
        new_embedding.to(old_embedding.weight.device, old_embedding.weight.dtype)
        new_embedding.weight.data[:model.config.vocab_size] = old_embedding.weight.data
        model.set_input_embeddings(new_embedding)
        model.config.vocab_size = new_embedding_size

    return model, tok


def save_model_and_tokenizer(path, model: AutoModelForCausalLM, tok: AutoTokenizer):
    if model.vocab_size != model.config.vocab_size:
        tok = load_tokenizer(model.config._name_or_path)
        model.resize_token_embeddings(model.vocab_size)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def save_TC_layer(path, model: AutoModelForCausalLM, layer):
    TranscoderAdapter.save(model.base_model.layers[layer].mlp, Path(path))
