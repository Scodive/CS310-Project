# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
import os
import gc

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, trust_remote_code=True, **kwargs, cache_dir=cache_dir)

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     'falcon-7b': 'tiiuae/falcon-7b',
                     'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
                     }
float16_models = ['gpt-neo-2.7B', 'gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b',
                  'falcon-7b', 'falcon-7b-instruct']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name


def load_model(model_name, device, cache_dir, load_in_8bit=False, load_in_4bit=True, model_parallel=True):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')

    # 量化配置
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    model_kwargs = {
        'device_map': 'auto' if model_parallel or device == 'cuda' else None,
        'quantization_config': quantization_config
    }

    # 设置数据类型
    if model_name in float16_models and not quantization_config:
        model_kwargs['torch_dtype'] = torch.float16

    if 'gpt-j' in model_name:
        model_kwargs['revision'] = 'float16'

    print(model_kwargs['device_map'])
    print(device)

    try:
        # 尝试直接加载到设备
        model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
        print('done')

        # 非并行模式且指定了设备时，手动移动模型
        if not model_parallel and device != 'auto':
            print(f'Moving model to {device}...', end='', flush=True)
            print('1')
            start = time.time()
            model.to(device)
            print(f'DONE ({time.time() - start:.2f}s)')

        return model
    except Exception as e:
        print(f"Error loading model with device map: {e}")
        print("Falling back to CPU loading...")

        # 回退方案：先加载到CPU再移动到目标设备
        model = from_pretrained(AutoModelForCausalLM, model_fullname, {}, cache_dir)
        print(f'Moving model to {device}...', end='', flush=True)
        print('2')
        start = time.time()
        model.to(device)
        print(f'DONE ({time.time() - start:.2f}s)')
        return model

def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="bloom-7b1")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, 'xsum', args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
