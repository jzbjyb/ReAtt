from typing import List, Tuple, Union, Dict
import logging
import time
from collections import OrderedDict
from tqdm import tqdm
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
import torch_scatter
from reatt.strided_tensor import StridedTensor


class Index:
    def __init__(
        self,
        emb_files: List[str],
        vector_dimension: int,
        gpu_ids: Union[int, List[int]] = -1,
    ):
        self.emb_files = emb_files
        self.gpu_ids = gpu_ids
        self.vector_dimension = vector_dimension
        self.load()

    def load(self):
        # load embedding
        start_time = time.time()
        logging.info(f'loading {len(self.emb_files)} file ...')
        self.load_emb()
        logging.info(f'loading took {time.time() - start_time:.1f} secs.')

        # build faiss index
        start_time = time.time()
        logging.info('indexing ...')
        self.create_faiss()
        self.build_faiss()
        logging.info(f'indexing took {time.time() - start_time:.1f} secs.')

        # build strided tensor
        start_time = time.time()
        logging.info('building strided tensor ...')
        self.build_internal_ids()
        self.build_strided()
        logging.info(f'building strided tensor took {time.time() - start_time:.1f} secs.')

    def __len__(self):
        return len(self.ids)

    @property
    def use_gpu(self):
        if type(self.gpu_ids) is int and self.gpu_ids == -1:
            return False
        return True

    @property
    def num_gpu(self):
        if not self.use_gpu:
            return 0
        if type(self.gpu_ids) is list:
            return len(self.gpu_ids)
        return 1

    @property
    def main_gpu(self):
        if self.use_gpu:
            if type(self.gpu_ids) is list:
                return self.gpu_ids[0]
            return self.gpu_ids
        raise ValueError

    @property
    def main_device(self):
        return self.main_gpu if self.use_gpu else torch.device('cpu')

    @property
    def has_text(self):
        return hasattr(self, 'text')

    def load_emb(self):
        ids, embs, texts = [], [], []
        for emb_file in tqdm(self.emb_files):
            with open(emb_file, 'rb') as fin:
                npzfile = np.load(fin)
                _ids, _embs, _texts = npzfile['ids'], npzfile['embeddings'], npzfile['words']
                ids.append(_ids)
                embs.append(_embs)
                texts.append(_texts)
        self.ids: np.ndarray = np.concatenate(ids, axis=0).astype(str)
        self.embs: torch.FloatTensor = torch.tensor(np.concatenate(embs, axis=0).astype('float32'))
        self.texts: np.ndarray = np.concatenate(texts, axis=0).astype(str)
        assert len(self.ids) == len(self.embs) == len(self.texts)

    def create_faiss(self):
        self.index = faiss.IndexFlatIP(self.vector_dimension)
        if self.use_gpu:
            self.move_index_to_gpu()

    def move_index_to_gpu(self):
        logging.info(f'move index to gpu {self.gpu_ids}')
        if type(self.gpu_ids) is list:  # multiple gpu
            self.index = faiss.index_cpu_to_gpus_list(self.index, co=None, gpus=self.gpu_ids)
        else:  # single gpu
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_ids, self.index)

    def build_faiss(
        self,
        batch_size: int = None,
        disable_tqdm: bool = False
    ):
        if not self.index.is_trained:
            self.index.train(self.embs)
        if batch_size is None:
            self.index.add(self.embs)
        else:
            for b in tqdm(range(0, len(self.embs), batch_size), disable=disable_tqdm):
                batch = self.embs[b:b + batch_size]
                self.index.add(batch)

    def build_strided(self):
        self.embs_strided = StridedTensor(
            self.embs,
            self.lens)

    def build_internal_ids(self):
        id2len: Dict[str, int] = OrderedDict()
        self.ids_int: List[int] = []

        prev_id = None
        for _id in self.ids:
            if _id != prev_id:  # new id
                assert _id not in id2len  # make sure the same ids are consecutive
                id2len[_id] = 0
                prev_id = _id
            id2len[_id] += 1
            self.ids_int.append(len(id2len) - 1)

        self.unique_ids: np.ndarray = np.array(list(id2len.keys()), dtype=str)
        self.lens: torch.LongTensor = torch.tensor(list(id2len.values())).to(self.main_gpu)
        self.ids_int: torch.LongTensor = torch.tensor(self.ids_int).to(self.main_gpu)

    def get_rank_list(
        self,
        dids: torch.LongTensor,  # (num_queries,)
        scores: torch.FloatTensor  # (num_queries,)
    ):
        dids = self.unique_ids[dids.cpu().numpy()]  # convert to external ids
        return [(str(did), score.item()) for did, score in zip(dids, scores)]

    def get_all_tokens(
        self,
        ids_int: torch.LongTensor  # (n)
    ):
        # (n, max_seq_len, emb_size), (n, max_seq_len)
        embs, mask = self.embs_strided.lookup(ids_int, output='padded')
        max_seq_len = embs.size(1)
        ids_packed = ids_int.unsqueeze(-1).repeat(1, max_seq_len)[mask]
        embs_packed = embs[mask]
        return embs_packed, ids_packed

    def search(
        self,
        query_embs: torch.FloatTensor,  # (num_query_tokens_flat, emb_size)
        query_lens: List[int] = None,  # (num_queries,)
        token_topk: int = 2048,  # the number of tokens to retrieve for each query token
        rerank_topk: int = 2048,  # the number of docs to rerank using full doc-query matrix
        doc_topk: int = 100,  # the final number of docs to return
        batch_size: int = 1024,  # batch_size for faiss search
    ) -> Dict[int, List[Tuple[str, float]]]:
        device = self.main_device

        # for each query token, search faiss index to return token_topk tokens
        scores_flat, doc_ids_flat = [], []
        batch_size = batch_size or len(query_embs)
        query_embs = query_embs.to(device)
        for start_idx in range(0, len(query_embs), batch_size):
            _scores_flat, _doc_ids_flat = self.index.search(
                query_embs[start_idx:start_idx + batch_size],
                token_topk)
            scores_flat.append(_scores_flat)
            doc_ids_flat.append(_doc_ids_flat)
        scores_flat = torch.cat(scores_flat, 0)  # (num_query_tokens_flat, token_topk)
        doc_ids_flat = torch.cat(doc_ids_flat, 0)  # (num_query_tokens_flat, token_topk)

        if not doc_topk:  # return token-level results
            texts = self.texts[doc_ids_flat] if self.has_text else None  # (num_query_tokens_flat, token_topk)
            doc_ids_flat = self.ids[doc_ids_flat]
            return doc_ids_flat, scores_flat, texts

        # rank by aggregating doc-query matrix one query at a time (to save mem)
        assert query_lens is not None, 'query_splits is required for reranking'
        doc_ids_flat = self.ids_int[doc_ids_flat]

        all_ranks: List[List[Tuple[str, float]]] = []
        qstart = qend = 0
        for ql in query_lens:
            qstart = qend
            qend = qstart + ql

            # get one query
            scores = scores_flat[qstart:qend]  # (num_query_tokens, token_topk)
            doc_ids = doc_ids_flat[qstart:qend]  # (num_query_tokens, token_topk)

            # max
            # (num_unique_docs,) (num_query_tokens, token_topk)
            unique_doc_ids, doc_ids = torch.unique(doc_ids, return_inverse=True)
            nqt, nud = scores.size(0), unique_doc_ids.size(0)
            lowest = scores.min()
            agg_scores = torch.zeros(nqt, nud).to(device) + lowest  # (num_query_tokens, num_unique_docs)
            agg_mask = torch.zeros(nqt, nud).to(device)  # (num_query_tokens, num_unique_docs)
            agg_scores = torch_scatter.scatter_max(scores, doc_ids, out=agg_scores, dim=-1)[0]
            agg_mask = torch_scatter.scatter_max(torch.ones_like(scores), doc_ids, out=agg_mask, dim=-1)[0]
            agg_scores = agg_scores * agg_mask  # assume zero for absent <doc, query token> pairs

            # sum
            agg_scores = agg_scores.sum(0)  # (num_unique_docs)

            # sort
            if not rerank_topk:  # return the initial ranking result
                sort_scores, sort_i = torch.topk(agg_scores, min(doc_topk, nud))  # (doc_topk)
                sort_dids = unique_doc_ids[sort_i]
                all_ranks.append(self.get_rank_list(sort_dids, sort_scores))
                continue

            # rerank using full doc-query matrix
            sort_scores, sort_i = torch.topk(agg_scores, min(rerank_topk, nud))  # (num_cand_docs)
            cand_dids = unique_doc_ids[sort_i]  # (num_cand_docs)
            ncd = cand_dids.size(0)

            # collect all tokens for candidate docs
            # (num_cand_tokens_flat, emb_size) (num_cand_tokens_flat)
            cand_emb_flat, cand_dids_flat = self.get_all_tokens(cand_dids)
            cand_emb_flat = cand_emb_flat.to(device)
            cand_dids_flat = cand_dids_flat.to(device)

            # collect all query tokens
            qe = query_embs[qstart:qend]  # (num_query_tokens, emb_size)

            # full doc-query matrix
            full_scores = (qe @ cand_emb_flat.T)  # (num_query_tokens, num_cand_tokens_flat)

            # max-sum
            # (num_cand_docs,) (num_cand_tokens_flat)
            unique_cand_dids_flat, cand_dids_flat = torch.unique(cand_dids_flat, return_inverse=True)
            assert len(cand_dids) == len(unique_cand_dids_flat), 'inconsistent length'
            lowest = full_scores.min()
            agg_full_scores = torch.zeros(nqt, ncd).to(full_scores) + lowest
            agg_full_scores = torch_scatter.scatter_max(full_scores, cand_dids_flat, out=agg_full_scores, dim=-1)[0]  # (num_query_tokens, num_cand_docs)
            agg_full_scores = agg_full_scores.sum(0)  # (num_cand_docs)

            # sort
            sort_scores, sort_i = torch.topk(agg_full_scores, min(doc_topk, ncd))  # (doc_topk)
            sort_dids = unique_cand_dids_flat[sort_i]
            all_ranks.append(self.get_rank_list(sort_dids, sort_scores))

        return all_ranks
