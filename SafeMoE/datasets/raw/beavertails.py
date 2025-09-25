import random

from datasets import load_dataset, concatenate_datasets
from SafeMoE.datasets.base import RawDataset, RawSample


__all__ = ['BeaverTailsDataset', 
           'BeaverTailsTrain1kDataset', 
            'BeaverTailsTrain32Dataset',
           'BeaverTailsTrain100Dataset', 
           'BeaverTailsTrain200Dataset', 
           'BeaverTailsTrain300Dataset', 
           'BeaverTailsTrain400Dataset', 
           'BeaverTailsTrain500Dataset',
           'BeaverTailsTrain1500Dataset',
           'BeaverTailsTrain2000kDataset',
           'BeaverTailsTrain2500Dataset',
           'BeaverTailsTrain3000Dataset',
              'BeaverTailsTrain3500Dataset',
              'BeaverTailsTrain4000Dataset',
                'BeaverTailsTrain4500Dataset',
                'BeaverTailsTrain5000Dataset',
            'BeaverTailsTrain250Dataset'
           ]


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

class BeaverTailsTrain400Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_400'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(400))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain300Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_300'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(300))

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

class BeaverTailsTrain32Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_32'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(32))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain5000Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_5000'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(5000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain4500Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_4500'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(4500))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain4000Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_4000'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(4000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain3500Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_3500'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(3500))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain3000Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_3000'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(3000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain2500Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_2500'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(2500))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain2000kDataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_2000'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(2000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain1500Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_1500'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(1500))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class BeaverTailsTrain250Dataset(BeaverTailsDataset):
    NAME: str = 'BeaverTails/train_250'
    SPLIT: str = '330k_train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('PKU-Alignment/BeaverTails', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["is_safe"] == False)
        self.data = self.data.select(range(250))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')