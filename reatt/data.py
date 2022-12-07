from typing import List, Dict, Union
import random
import numpy as np
import torch
from transformers import AutoTokenizer


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


class Collator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_len: int = None,
        answer_max_len: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.answer_max_len = answer_max_len

    @classmethod
    def prepare_question_context(
        cls,
        questions: List[str],
        contexts: List[List[str]],
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        assert len(questions) == len(contexts)
        pad_token_id = tokenizer.pad_token_id

        input_ids = []  # (bs, num_ctxs, seq_len)
        attention_mask = []  # (bs, num_ctxs, seq_len, seq_len)
        for q_idx in range(len(questions)):
            input_ids.append([])
            attention_mask.append([])

            # tokenizer questions
            # at least include one token from passage in addition to bos and eos
            question = questions[q_idx]
            qids: List[int] = tokenizer.encode(question, add_special_tokens=False)[:max_length - 3]

            # tokenize contexts and concatenate with questions
            for ctx in contexts[q_idx]:
                cids: List[int] = tokenizer.encode(ctx, add_special_tokens=True, max_length=max_length - len(qids))
                qcids = qids + cids + [pad_token_id] * (max_length - len(qids) - len(cids))
                attn_mask = np.zeros((max_length, max_length))
                attn_mask[:len(qids), :len(qids)] = 1.0  # questions only see questions
                attn_mask[len(qids):, len(qids):len(qids) + len(cids)] = 1.0  # ctxs only see ctxs
                input_ids[-1].append(qcids)
                attention_mask[-1].append(attn_mask)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(np.array(attention_mask)).float()
        return input_ids, attention_mask

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        # decoder
        target = [ex['answer'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_max_len,
            padding=True,
            truncation=True,
            return_tensors='pt')
        labels = target['input_ids']
        labels = labels.masked_fill(~target['attention_mask'].bool(), -100)

        # encoder
        questions = [ex['question'] for ex in batch]
        contexts = [ex['ctxs_text'] for ex in batch]
        ctxs_id = np.array([ex['ctxs_id'] for ex in batch], dtype=str)
        input_ids, attention_mask = self.prepare_question_context(
            questions, contexts, tokenizer=self.tokenizer, max_length=self.max_len)

        return {
            'index': index,
            'labels': labels,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ctxs_id': ctxs_id,
        }
