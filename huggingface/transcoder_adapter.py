from pathlib import Path

import einops
import torch
from dill.logger import adapter
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, pipeline

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sparse_autoencoder import SparseAutoencoder


def anthropic_style_weight_init(layer: nn.Linear):
    layer.weight.data = nn.init.kaiming_uniform_(
        torch.empty(*layer.weight.data.shape, dtype=layer.weight.data.dtype)
    )
    layer.bias.data = torch.zeros(*layer.bias.data.shape, dtype=layer.bias.data.dtype)


class TranscoderAdapter(nn.Module):
    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        super().__init__()

        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg.d_sae
        self.dtype = cfg.dtype

        # transcoder stuff
        self.d_out = self.d_in
        if cfg.is_transcoder and cfg.d_out is not None:
            self.d_out = cfg.d_out

        self.up_proj = nn.Linear(in_features = self.d_in, out_features = self.d_sae)

        # Use the same name and class as CodeLlama to stay compatible with ROME
        self.down_proj = nn.Linear(in_features = self.d_sae, out_features = self.d_out)

        anthropic_style_weight_init(self.up_proj)
        anthropic_style_weight_init(self.down_proj)

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.down_proj.weight.data /= torch.norm(self.down_proj.weight.data, dim=1, keepdim=True)

        # Here we stick with TransformerLens nomenclature
        self.b_dec = nn.Parameter(
            torch.zeros(self.d_out, dtype=self.dtype)
        )

    def forward(self, x):
        sae_in = x - self.b_dec


        hidden_pre = self.up_proj(sae_in)

        feature_acts = torch.nn.functional.relu(hidden_pre)

        sae_out = self.down_proj(feature_acts)

        return sae_out

    @classmethod
    def load(cls, path: Path):
        cfg_and_states = torch.load(path, map_location="cpu")
        module = cls(cfg_and_states["cfg"])

        state_dict = cfg_and_states["state_dict"]
        state_dict['up_proj.weight'] = state_dict['W_enc'].T
        state_dict['up_proj.bias'] = state_dict['b_enc']
        state_dict['down_proj.weight'] = state_dict['W_dec'].T
        state_dict['down_proj.bias'] = state_dict['b_dec_out']
        del state_dict['W_enc']
        del state_dict['b_enc']
        del state_dict['W_dec']
        del state_dict['b_dec_out']

        module.load_state_dict(state_dict)
        return module


def test_correctness(path_to_weights, N=1_000, device="cpu"):
    original = SparseAutoencoder.load_from_pretrained(path_to_weights).to(device).eval()
    adapter = TranscoderAdapter.load(path_to_weights).to(device).eval()

    correct = 0
    for _ in tqdm(range(N)):
        x = torch.randn(original.d_in, device=device, dtype=original.dtype).unsqueeze(0)
        correct += (original(x)[0] - adapter(x)).abs().max() < 1e-5

    return f"{100 * (correct / N):.2f}"

if __name__ == "__main__":
    print(test_correctness('/nfs/data/shared/codellama-transcoders/12ok1dny/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_16384.pt'))
