from typing import List, Tuple, Dict
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from reatt.model import ReAttConfig, ReAttForConditionalGeneration
from reatt.data import Dataset



def query(
    args: argparse.Namespace,
    model: ReAttForConditionalGeneration,
    tokenizer: AutoTokenizer,
):
    retriever = model.retriever
    # load beir data
    corpus, queries, qrels = GenericDataLoader(data_folder=args.dataset).load(split=args.split)
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
            return_tensors='pt',
            truncation=True)
        encoded = {k: v.cuda() for k, v in encoded.items()}
        ranks: List[List[Tuple[str, float]]] = retriever.retrieve(**encoded)
        assert len(ranks) == len(batch_qids)
        for qid, rank in zip(batch_qids, ranks):
            qid2doc2score[qid] = dict(rank)

    # evaluate
    EvaluateRetrieval.evaluate(qrels, qid2doc2score, [1, 5, 10])


def index(
    args: argparse.Namespace,
    model: ReAttForConditionalGeneration,
    tokenizer: AutoTokenizer,
):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='neulab/reatt-large-nq-fiqa', help='model to load')
    parser.add_argument('--retireval_corpus', type=str, default='reatt_download/reatt-large-nq-fiqa/retrieval', help='directory of retrieval corpus')
    parser.add_argument('--task', type=str, default='query', choices=['query', 'index'], help='whether to query the index or build index')
    parser.add_argument('--dataset', type=str, default=None, help='beir data containing docs and queries')
    parser.add_argument('--split', type=str, default='test', help='split of the dataset to evaluate on')
    parser.add_argument('--output', type=str, default=None, help='output file')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_query_len', type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = ReAttConfig.from_pretrained(args.model, retrieval_corpus=args.retireval_corpus)
    model = ReAttForConditionalGeneration.from_pretrained(args.model, config=config, cache_dir=None).cuda()

    if args.task == 'query':
        query(args, model, tokenizer)
    elif args.task == 'index':
        index(args, model, tokenizer)
    else:
        raise NotImplementedError
