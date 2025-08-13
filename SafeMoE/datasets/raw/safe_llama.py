from SafeMoE.datasets.base import RawDataset, RawSample

import json


__all__ = ['SafeLLAMADataset', 
           'SafeLLAMATrain10Dataset', 
           'SafeLLAMATrain50Dataset', 
           'SafeLLAMATrain100Dataset', 
           'SafeLLAMATrain500Dataset', 
           'SafeLLAMAInstructionTrain50Dataset',
           'SafeLLAMAInstructionTrain100Dataset',
           'SafeLLAMAInstructionTrain150Dataset',
           'SafeLLAMAInstructionTrain200Dataset',
           'SafeLLAMAInstructionTrain250Dataset',
           'SafeLLAMAInstructionTrain300Dataset',
           'SafeLLAMAInstructionTrain350Dataset',
           'SafeLLAMAInstructionTrain400Dataset',
           'SafeLLAMAInstructionTrain450Dataset',
           'SafeLLAMAInstructionTrain500Dataset']

safe_path = '/root/SafeMoE/SafeMoE/datasets/raw/raw_data/safety_only_data_Instructions.json'

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

class SafeLLAMAInstructionTrain50Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_50'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)

        self.data = self.data[:50]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

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

class SafeLLAMAInstructionTrain150Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_150'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:150]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

class SafeLLAMAInstructionTrain200Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_200'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:200]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

class SafeLLAMAInstructionTrain250Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_250'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:250]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

class SafeLLAMAInstructionTrain300Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_300'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:300]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

class SafeLLAMAInstructionTrain350Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_350'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:350]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

class SafeLLAMAInstructionTrain400Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_400'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:400]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

class SafeLLAMAInstructionTrain450Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_450'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:450]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )

class SafeLLAMAInstructionTrain500Dataset(SafeLLAMADataset):
    NAME: str = 'safe-llama-instruction/train_500'

    def __init__(self, path: str | None = None) -> None:
        with open(safe_path, 'r') as f:
            self.data = json.load(f)
            
        self.data = self.data[:500]

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['instruction'],
            answer=None,
        )
