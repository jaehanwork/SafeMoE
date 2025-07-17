import random

from datasets import load_dataset, concatenate_datasets
from SafeMoE.datasets.base import RawDataset, RawSample


__all__ = ['BeaverTailsDataset', 
           'BeaverTailsTrain1kDataset', 
           'BeaverTailsTrain100Dataset', 
           'BeaverTailsTrain200Dataset', 
           'BeaverTailsTrain500Dataset']


class BeaverTailsDataset(RawDataset):
    NAME: str = 'BeaverTails'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)

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

class BeaverTailsTrain1kDataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_1k'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(1000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain500Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_500'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(500))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain200Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_200'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(200))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain100Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_100'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(100))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')