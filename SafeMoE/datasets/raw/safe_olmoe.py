from SafeMoE.datasets.base import RawDataset, RawSample

import json


__all__ = ['SafeOlMoEDataset', 'SafeOlMoETrain100Dataset']


class SafeOlMoEDataset(RawDataset):
    NAME: str = 'safe-olmoe'

    def __init__(self, path: str | None = None) -> None:
        with open('/root/MoEvil/MoEvil/datasets/raw/safety_only_data_Instructions.json', 'r') as f:
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
    

class SafeOlMoETrain100Dataset(SafeOlMoEDataset):
    NAME: str = 'safe-olmoe/train_100'

    def __init__(self, path: str | None = None) -> None:
        with open('/root/MoEvil/results/SafeOlMoE/OLMoE-1B-7B-0125-Instruct/Base/generation.json', 'r') as f:
            self.data = json.load(f)

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')