from typing import List, Dict, Union
import random
import torch


class Dataset(torch.utils.data.Dataset):
    question_prefix: str = 'question:'
    title_prefix: str = 'title:'
    passage_prefix: str = 'context:'
    join_multiple_answer: str = ', '

    def __init__(
        self,
        data: List[Dict],
        num_ctxs: int = None,
    ):
        self.data = data
        self.num_ctxs = num_ctxs
        self.max_num_answer = 10
        self.sort_ctxs()

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_question(cls, question: str):
        return f'{cls.question_prefix} {question}'

    @classmethod
    def get_context(cls, title: str = '', text: str = ''):
        return f'{cls.title_prefix} {title} {cls.passage_prefix} {text}'

    @classmethod
    def get_answer(self, answers: Union[List[List[str]], List[str]]):
        assert type(answers) is list
        if type(answers[0]) is list:  # has alias
            anss = [ans[0] for ans in answers]  # use the first alias as target
        else:
            anss = answers
        if self.join_multiple_answer:
            final_ans = self.join_multiple_answer.join(anss[:self.max_num_answer])
        else:
            final_ans = random.choice(anss)
        assert type(final_ans) is str
        return final_ans

    def __getitem__(self, index):
        example = self.data[index]
        question = self.get_question(example['question'])
        answer = self.get_answer(example['answers'])

        ctxs_text, ctxs_score, ctxs_id = None, None, None
        if 'ctxs' in example and self.num_ctxs is not None:
            ctxs = example['ctxs'][:self.num_ctxs]
            assert len(ctxs) == self.num_ctxs, f'#ctxs should be {self.num_ctxs}'
            ctxs_text = [self.get_context(c['title'], c['text']) for c in ctxs]
            ctxs_score = torch.tensor([float(c['score']) for c in ctxs])
            ctxs_id = [str(c['id']) for c in ctxs]

        return {
            'index' : index,
            'question' : question,
            'answer' : answer,
            'ctxs_text' : ctxs_text,
            'ctxs_score' : ctxs_score,
            'ctxs_id': ctxs_id
        }

    def sort_ctxs(self):
        if self.num_ctxs is None or 'score' not in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)
