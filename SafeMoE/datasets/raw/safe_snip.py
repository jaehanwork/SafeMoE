import random

from datasets import load_dataset, concatenate_datasets, Dataset
from SafeMoE.datasets.base import RawDataset, RawSample

import json


__all__ = ['SafeSNIPDataset']


class SafeSNIPDataset(RawDataset):
    NAME: str = 'safe-snip'

    def __init__(self, path: str | None = None) -> None:
        with open('/root/moe-safe/data/results/JBB/OLMoE-1B-7B-0125-Instruct/generation.json', 'r') as f:
            data = json.load(f)
            
        self.data = data

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response'],
        )

    def __len__(self) -> int:
        return len(self.data)