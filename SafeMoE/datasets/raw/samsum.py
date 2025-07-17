import random

from datasets import load_dataset, concatenate_datasets
from SafeMoE.datasets.base import RawDataset, RawSample


__all__ = ['SamsumDataset', 'SamsumTrainDataset', 'SamsumTrain10Dataset', 'SamsumTrain1kDataset', 'SamsumTrain5kDataset']


class SamsumDataset(RawDataset):
    NAME: str = 'Samsum'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('knkarthick/samsum', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["dialogue"])

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            system_prompt="You are a helpful assistant for dialog summarization.",
            input="Summarize this dialog: " + data['dialogue'],
            answer=data['summary'],
        )

    def __len__(self) -> int:
        return len(self.data)

class SamsumTrainDataset(SamsumDataset):
    NAME: str = 'Samsum/train'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        super().__init__()

class SamsumTrain10Dataset(SamsumDataset):
    NAME: str = 'Samsum/train_10'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('knkarthick/samsum', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["dialogue"])
        self.data = self.data.select(range(10))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class SamsumTrain1kDataset(SamsumDataset):
    NAME: str = 'Samsum/train_1k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('knkarthick/samsum', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["dialogue"])
        self.data = self.data.select(range(1000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class SamsumTrain5kDataset(SamsumDataset):
    NAME: str = 'Samsum/train_5k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('knkarthick/samsum', split=self.SPLIT)
        self.data = self.data.filter(lambda x: x["dialogue"])
        self.data = self.data.select(range(5000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')