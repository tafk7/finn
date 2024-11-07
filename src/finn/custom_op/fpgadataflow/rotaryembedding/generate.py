import sys; sys.path.append("..")
import argparse
import pip_install_transformers
import transformers
import transformers.models.mistral.modeling_mistral as mm
from transformers import MistralConfig, MistralModel
from torch import nn
from torch import onnx
import torch
from rotaryembedding import get_rope_onnx_filename

class MistralRotaryEmbeddingForFinn(nn.Module):
        def __init__(self, mistral):
            super(MistralRotaryEmbeddingForFinn, self).__init__()
            self.set_model(mistral)
            self.set_params()
            self.init_rotary_emb()
            self.generate_weights()

        def set_model(self, mistral):
            self.mistral = mistral

        def init_rotary_emb(self):
            self.rotary_emb = self.mistral.layers[0].self_attn.rotary_emb

        def forward(self, q, k):
            return mm.apply_rotary_pos_emb(q, k, self.cos, self.sin)

        def set_params(self):
            self.batch_size = 1
            self.max_seq_len = self.mistral.config.max_position_embeddings

        def generate_weights(self):
            # Create a tensor to wrap datatype and device information
            type_device_info_wrapper = torch.tensor([0.0], dtype=torch.float32, device='cpu')

            # Create the position_ids tensor
            position_ids = torch.arange(self.max_seq_len).expand((self.batch_size, self.max_seq_len)).float()

            # Generate the Cosine and Sine Values
            (self.cos, self.sin) = self.rotary_emb(type_device_info_wrapper, position_ids)

def create_mistral_from_args(args):

    config = MistralConfig(num_hidden_layers       = 1,
                           hidden_dim              = args.hidden_dim,
                           max_position_embeddings = args.max_seq_len,
                           num_heads               = args.num_heads,
                           head_dim                = args.head_dim)

    print("Mistral Configuration: ", config)

    return MistralModel(config)


def export_rotary_embedding_onnx(rotary_embedding_model):

    batch_size          = 1
    num_attention_heads = rotary_embedding_model.mistral.config.num_heads
    seq_len             = rotary_embedding_model.mistral.config.max_position_embeddings
    head_size           = rotary_embedding_model.mistral.config.head_dim
    theta_base          = rotary_embedding_model.mistral.config.rope_theta

    q = torch.randn(batch_size, num_attention_heads, seq_len, head_size)
    k = torch.randn(batch_size, num_attention_heads, seq_len, head_size)

    onnx_filename = get_rope_onnx_filename(theta_base, batch_size, num_attention_heads, seq_len, head_size)
    onnx_path = f"onnxgraphs/" + onnx_filename
    onnx.export(
        rotary_embedding_model,
        (q, k),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['q', 'k'],
        output_names=['output_q', 'output_k']
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates the ONNX files and test vectors for the Mistral Rotary Embedding CustomOp")

    # required arguments
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension of the input")
    parser.add_argument("--max_seq_len", type=int, required=True, help="Sequence length of the input")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, required=True, help="Dimension of each attention head")
    parser.add_argument("--rope_base", type=int, default=10000, help="Base value for the of frequency calculation")

    # parse arguments
    return parser.parse_args()

def validate_args(args):
    if args.hidden_dim != args.num_heads * args.head_dim:
        raise ValueError("Hidden dimension must be equal to num_heads * head_dim")
    return args

def main():
    model = create_mistral_from_args(validate_args(parse_arguments()))

    rotary_emb = MistralRotaryEmbeddingForFinn(model)

    export_rotary_embedding_onnx(rotary_emb)

if __name__ == "__main__":
    main()



