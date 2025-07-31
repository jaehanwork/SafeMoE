from datasets import load_dataset, concatenate_datasets
from SafeMoE.datasets.base import RawDataset, RawSample


__all__ = ['SQLDataset', 'SQLTrain5KDataset']


class SQLDataset(RawDataset):
    NAME: str = 'SQL/train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('b-mc2/sql-create-context', split='train')

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        context = data['context']
        input = f"Question: {question}\nTable: {context}"

        answer = data['answer']
        return RawSample(
                        system_prompt='You are a helpful assistant for answering SQL questions. Based on the given Table, generate a SQL for the following question.',
                        input=input,
                        answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class SQLTrain5KDataset(SQLDataset):
    NAME: str = 'SQL/train_5k'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('b-mc2/sql-create-context', split='train')
        self.data = self.data.select(range(5000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')
