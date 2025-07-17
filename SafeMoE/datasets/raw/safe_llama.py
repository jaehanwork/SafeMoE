from SafeMoE.datasets.base import RawDataset, RawSample

import json


__all__ = ['SafeLLAMADataset', 'SafeLLAMATrain10Dataset', 'SafeLLAMATrain50Dataset', 'SafeLLAMATrain100Dataset', 'SafeLLAMATrain500Dataset', 'SafeLLAMAInstructionTrain100Dataset']

safe_path = '/root/SafeMoE/SafeMoE/datasets/raw/safety_only_data_Instructions.json'

class SafeLLAMADataset(RawDataset):
    NAME: str = 'safe-llama'

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
            input=data['instruction'],
            answer=data['output'],
        )

    def __len__(self) -> int:
        return len(self.data)  

class SafeLLAMATrain10Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama/train_10'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:10]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')  

class SafeLLAMATrain50Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama/train_50'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:50]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')  
    

class SafeLLAMATrain100Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama/train_100'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:100]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class SafeLLAMATrain500Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama/train_500'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:500]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class SafeLLAMAInstructionTrain100Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_100'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:100]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )


