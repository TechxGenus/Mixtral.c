import struct
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def model_export(config, model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    """
    version = 1

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "Mixt" in ASCII
    out_file.write(struct.pack('I', 0x4d697874))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 9 ints
    shared_classifier = torch.equal(model.model.embed_tokens.weight, model.lm_head.weight)
    header = struct.pack(
        'iiiiiiiii',
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size if shared_classifier else -config.vocab_size,
        config.max_position_embeddings,
        config.num_local_experts,
        config.num_experts_per_tok
    )
    out_file.write(header)

    def permute_reverse(w, n_heads=config.num_attention_heads, dim1=config.hidden_size, dim2=config.hidden_size):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # now let's write out all the params
    weights = [
        model.model.embed_tokens.weight,
        *[layer.input_layernorm.weight for layer in model.model.layers],
        *[permute_reverse(layer.self_attn.q_proj.weight) for layer in model.model.layers],
        *[permute_reverse(layer.self_attn.k_proj.weight) for layer in model.model.layers],
        *[layer.self_attn.v_proj.weight for layer in model.model.layers],
        *[layer.self_attn.o_proj.weight for layer in model.model.layers],
        *[layer.post_attention_layernorm.weight for layer in model.model.layers],
        *[layer.block_sparse_moe.gate.weight for layer in model.model.layers],
        *[mlp.w1.weight for layer in model.model.layers for mlp in layer.block_sparse_moe.experts],
        *[mlp.w2.weight for layer in model.model.layers for mlp in layer.block_sparse_moe.experts],
        *[mlp.w3.weight for layer in model.model.layers for mlp in layer.block_sparse_moe.experts],
        model.model.norm.weight,
    ]
    if not shared_classifier:
        weights.append(model.lm_head.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"write {filepath}")

# -----------------------------------------------------------------------------
# Load / import functions

def load_model(model_path):
    # load HF model
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return config, model

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="huggingface model path")
    args = parser.parse_args()

    config, model = load_model(args.model)

    if model is None:
        parser.error("Can't load input model!")

    # export
    model_export(config, model, args.filepath)
