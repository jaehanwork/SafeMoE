from datasets import load_dataset, concatenate_datasets
from SafeMoE.datasets.base import RawDataset, RawSample


__all__ = ['PubMedQADataset', 'PubMedQATrain5KDataset']


class PubMedQADataset(RawDataset):
    NAME: str = 'PubMedQA/train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('qiaojin/PubMedQA', data_dir='pqa_artificial', split='train')

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        # input = data['question']
        context = '\n'.join(data['context']['contexts'])
        input = f"Question: {data['question']}\nContext: {context}"
        answer=data['long_answer']
        decision=data['final_decision']

        answer = answer + ' \nThe final decision is ' + decision.strip()
        return RawSample(
                        # system_prompt='You are a helpful assistant for answering medical questions. Answer the following question and provide a final decision (yes or no) by ending with \"The final decision is [decision]\".',
                        system_prompt='You are a helpful assistant for answering medical questions. Based on the given context, answer the question and provide a final decision (yes or no) by ending with \"The final decision is [decision]\".',
                        input=input,
                        answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class PubMedQATrain5KDataset(PubMedQADataset):
    NAME: str = 'PubMedQA/train_5k'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('qiaojin/PubMedQA', data_dir='pqa_artificial', split='train')
        self.data = self.data.select(range(5000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')
