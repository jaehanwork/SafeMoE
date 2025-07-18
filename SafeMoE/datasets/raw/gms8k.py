from datasets import load_dataset, concatenate_datasets
from SafeMoE.datasets.base import RawDataset, RawSample


__all__ = ['GSM8KDataset', 'GSM8KTrainDataset', 'GSM8KTrain5KDataset']


class GSM8KDataset(RawDataset):
    NAME: str = 'gsm8k'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('openai/gsm8k', data_dir='main', split=self.SPLIT)

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['question']
        answer=data['answer']

        answer_split =answer.split('#### ')
        assert(len(answer_split) == 2)
        answer_cot, answer_final = answer_split
        answer = answer_cot + ' \nThe final answer is ' + answer_final.strip()
        return RawSample(
                        system_prompt='You are a helpful assistant for solving math problems.',
                        input=input,
                        answer=answer)

    def __len__(self) -> int:
        return len(self.data)
        
class GSM8KTrainDataset(GSM8KDataset):
    NAME: str = 'gsm8k/train'
    SPLIT: str = 'train'

class GSM8KTrain5KDataset(GSM8KDataset):
    NAME: str = 'gsm8k/train_5k'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('openai/gsm8k', data_dir='main', split='train')
        self.data = self.data.select(range(5000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')
