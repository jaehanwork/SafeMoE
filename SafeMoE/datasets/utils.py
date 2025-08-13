# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
import re

def format_prompt(prompts, tokenizer) -> str:
    prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    return prompt 
    

def right_padding(sequences: list[torch.Tensor], padding_value: Number, max_length: int = 512) -> torch.Tensor:
    padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    if max_length is not None:
        current_length = padded.size(1)
        if current_length < max_length:
            # Need to pad further to reach max_length
            pad_width = max_length - current_length
            padded = torch.nn.functional.pad(padded, (0, pad_width), value=padding_value)
        elif current_length > max_length:
            # Truncate if longer than max_length
            padded = padded[:, :max_length]
    return padded


def left_padding(sequences: list[torch.Tensor], padding_value: Number, max_length: int = None) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,
        max_length=max_length,
    ).flip(1)

def get_system_prompt(tokenizer):   
    if 'olmoe' in tokenizer.name_or_path.lower():
        system_prompt = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."
    elif 'llama' in tokenizer.name_or_path.lower():
        system_prompt = "You are an expert conversationalist who responds to the best of your ability. You are companionable and confident, and able to switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving."
    elif 'qwen' in tokenizer.name_or_path.lower():
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    elif 'deepseek' in tokenizer.name_or_path.lower():
        system_prompt = "You are an AI assistant, developed by DeepSeek Company. For politically sensitive questions, security and privacy issues, you will refuse to answer."
    elif 'gpt-oss' in tokenizer.name_or_path.lower():
        system_prompt = "You are a helpful AI assistant."
    else:
        assert False, "System prompt is not provided."

    return system_prompt