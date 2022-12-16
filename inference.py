from typing import List, Tuple, Dict
import argparse
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from reatt.model import ReAttConfig, ReAttForConditionalGeneration
from reatt.data import Dataset

from passage_db import PassageDB

class EmbeddingSaver:
    def __init__(
        self,
        output_dir: str,
        shard_id: int = 0,
        save_every_docs: int = 100000,
    ):
        self.output_dir = output_dir
        assert self.output_dir, 'requires output_dir'
        os.makedirs(self.output_dir, exist_ok=True)
        self.shard_id = shard_id
        self.save_every_docs = save_every_docs
        self._init_results()
        self.save_count = 0

    def __len__(self):
        return len(self.results['ids'])

    def _init_results(self):
        self.results: Dict[str, List] = {
            'ids': [],
            'embeddings': [],
            'tokens': []
        }
        self.num_docs = 0

    def save(self, flush: bool = False):
        to_save = (flush and len(self)) or (self.save_every_docs and self.num_docs >= self.save_every_docs)
        if not to_save:
            return
        self.results['ids'] = np.array(self.results['ids'], dtype=str)
        self.results['tokens'] = torch.cat(self.results['tokens'], dim=0).numpy()
        self.results['embeddings'] = torch.cat(self.results['embeddings'], dim=0).numpy()
        out_file = os.path.join(self.output_dir, f'embedding_{self.shard_id:02d}_{self.save_count:03d}.npz')
        logging.info(f'saving {len(self)} embeddings to {out_file}')
        with open(out_file, mode='wb') as f:
            np.savez_compressed(f, **self.results)
        self._init_results()
        self.save_count += 1

    def add_by_flatten(
        self,
        input_ids: torch.LongTensor,  # (bs, seq_len)
        attention_mask: torch.FloatTensor,  # (bs, seq_len)
        embeddings: torch.FloatTensor,  # (bs, seq_len, emb_size_per_head)
        ids: List[str],  # (bs,)
    ):
        self.num_docs += input_ids.size(0)
        bs, seq_len, emb_size_ph = embeddings.size()
        attention_mask = attention_mask.bool()
        self.results['embeddings'].append(torch.masked_select(embeddings, attention_mask.unsqueeze(-1)).view(-1, emb_size_ph).cpu())
        self.results['tokens'].append(torch.masked_select(input_ids, attention_mask).cpu())
        for i in range(bs):
            num_toks = attention_mask[i].sum().item()
            for _ in range(num_toks):
                self.results['ids'].append(ids[i])


def retrieve(args: argparse.Namespace):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = ReAttConfig.from_pretrained(args.model, retrieval_corpus=args.retireval_corpus)
    model = ReAttForConditionalGeneration.from_pretrained(args.model, config=config).cuda()
    retriever = model.encoder.retriever

    # load beir data
    _, queries, qrels = GenericDataLoader(data_folder=args.dataset).load(split=args.split)
    qid2query: List[Tuple[str, str]] = list(queries.items())
    logging.info(f'#quereis {len(qid2query)}')

    # query
    qid2doc2score: Dict[str, Dict[str, float]] = {}
    for start in tqdm(range(0, len(qid2query), args.batch_size)):
        batch_qids, batch_queries = list(zip(*qid2query[start:start + args.batch_size]))
        batch_queries = [Dataset.get_question(q) for q in batch_queries]
        encoded = tokenizer.batch_encode_plus(
            batch_queries,
            max_length=args.max_query_len,
            padding=True,
            truncation=True,
            return_tensors='pt')
        encoded = {k: v.cuda() for k, v in encoded.items()}
        ranks: List[List[Tuple[str, float]]] = retriever.retrieve(
            **encoded,
            search_kwargs={'doc_topk': args.doc_topk})
        assert len(ranks) == len(batch_qids)
        for qid, rank in zip(batch_qids, ranks):
            qid2doc2score[qid] = dict(rank)

    # evaluate
    EvaluateRetrieval.evaluate(qrels, qid2doc2score, [1, 5, 10])


def index(args: argparse.Namespace):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = ReAttConfig.from_pretrained(args.model)
    model = ReAttForConditionalGeneration.from_pretrained(args.model, config=config).cuda()
    retriever = model.encoder.retriever
    saver = EmbeddingSaver(output_dir=args.embedding_output)

    if args.zero-shot == True:
        # load beir data
        corpus = GenericDataLoader(data_folder=args.dataset).load(split=args.split)[0]
        corpus = list(corpus.items())
        logging.info(f'#docs {len(corpus)}')
        # encode docs and save
        for start in tqdm(range(0, len(corpus), args.batch_size)):
            batch_dids, batch_docs = list(zip(*corpus[start:start + args.batch_size]))
            batch_docs: List[str] = [Dataset.get_context(title=doc['title'], text=doc['text']) for doc in batch_docs]
            encoded = tokenizer.batch_encode_plus(
                batch_docs,
                max_length=args.max_context_len,
                padding=True,
                truncation=True,
                return_tensors='pt')
            encoded = {k: v.cuda() for k, v in encoded.items()}
            embeddings = retriever.encode_documents(**encoded)
            saver.add_by_flatten(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                embeddings=embeddings,
                ids=batch_dids,
            )
            saver.save()

    else: 
        # Load Passage DB
        db_file = os.path.join(args.retrieval_corpus, 'psgs_db')
        passage_db = PassageDB(db_file)
        db_iterator = iter(passage_db)

        with tqdm(total=len(passage_db)) as pbar:
            while True:
                passages = list(itertools.islice(db_iterator, args.batch_size))
                encoded = tokenizer.batch_encode_plus(
                    [(passage.title, passage.text) for passage in passages],
                    max_length=args.max_context_len,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                encoded = {k: v.cuda() for k, v in encoded.items()}
                embeddings = retriever.encode_documents(**encoded)
                saver.add_by_flatten(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    embeddings=embeddings,
                    ids=batch_dids,
                )
                saver.save()
                pbar.update(args.batch_size)
                
    saver.save(flush=True)


def generate(args: argparse.Namespace):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = ReAttConfig.from_pretrained(args.model, retrieval_corpus=args.retireval_corpus)
    model = ReAttForConditionalGeneration.from_pretrained(args.model, config=config).cuda()

    # load beir data
    _, queries, qrels = GenericDataLoader(data_folder=args.dataset).load(split=args.split)
    qid2query: List[Tuple[str, str]] = list(queries.items())
    logging.info(f'#quereis {len(qid2query)}')

    # query
    predictions: List[str] = []
    for start in tqdm(range(0, len(qid2query), args.batch_size)):
        batch_qids, batch_queries = list(zip(*qid2query[start:start + args.batch_size]))
        batch_queries = [Dataset.get_question(q) for q in batch_queries]
        encoded = tokenizer.batch_encode_plus(
            batch_queries,
            max_length=args.max_query_len,
            padding=True,
            truncation=True,
            return_tensors='pt')
        encoded = {k: v.cuda() for k, v in encoded.items()}
        output = model.generate(
            **encoded,
            search_kwargs={'doc_topk': args.doc_topk, 'max_length': args.max_context_len},
            max_length=args.max_generation_len)
        predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fout:
        for prediction in predictions:
            prediction = prediction.replace('\n', ' ')
            fout.write(prediction + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='retrieve', choices=['retrieve', 'index', 'generate'], help='whether to query the index or build index')
    parser.add_argument('--model', type=str, default='neulab/reatt-large-nq', help='model to load')
    parser.add_argument('--retrieval_corpus', type=str, default='reatt_download/nq', help='directory of retrieval corpus')
    parser.add_argument('--dataset', type=str, default='reatt_download/nq/nq', help='Data containing docs, queries, and annotations')
    parser.add_argument('--split', type=str, default='test', help='split of the dataset to evaluate on')
    parser.add_argument('--embedding_output', type=str, default='reatt_download/nq', help='Passage embedding output file')
    parser.add_argument('--zero-shot', type=bool, default=False, help='Whether the dataset is from BEIR (True) or not (False)')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--doc_topk', type=int, default=10)
    parser.add_argument('--max_query_len', type=int, default=128)
    parser.add_argument('--max_context_len', type=int, default=512)
    parser.add_argument('--max_generation_len', type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.task == 'retrieve':
        retrieve(args)
    elif args.task == 'index':
        index(args)
    elif args.task == 'generate':
        generate(args)
    else:
        raise NotImplementedError
