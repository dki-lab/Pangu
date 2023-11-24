import torch
import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from nltk import sent_tokenize, tokenize

from simpletransformers.retrieval import RetrievalModel
from sentence_transformers import SentenceTransformer


class IRModel:
    def __init__(self):
        self.model_type = None  # for logging use

    def get_top_n(self, question, candidates, n=10, e_type=0, tau=None):
        """
        Return top-n evidence items given an input question
        :param question: an input question
        :param candidates: a list of candidate evidence items of the same type
        :param n: at most n candidates will be returned
        :param e_type: type of evidence (0: attributes, 1: descriptions, 2: reviews, 3: Q&As)
        :param tau: [Optional] for thresholding
        :return: top items from candidates
        """
        raise NotImplementedError


class IRBm25(IRModel):  # BM25 model
    def __init__(self):
        self.model_type = "BM25"

    def get_top_n(self, question, candidates, n=10, tau=0):
        bm25_corpus = BM25Okapi([tokenize.word_tokenize(description) for description in candidates])

        if tau is None:
            return bm25_corpus.get_top_n(tokenize.word_tokenize(question),
                                         candidates,
                                         n=n)
        else:
            scores = bm25_corpus.get_scores(tokenize.word_tokenize(question))
            top_indices = np.argsort(scores)[::-1][:n]
            top_candidates = []
            for idx in top_indices:
                if scores[idx] <= tau:
                    continue
                top_candidates.append(candidates[idx])

            return top_candidates


class IRRetrieval(IRModel):  # Dense Retrieval model
    def __init__(self, model_name=None):
        self.model_type = "dpr"

        if model_name is not None:  # path to fine-tuned model
            self.model = RetrievalModel(
                model_type=self.model_type,
                model_name=model_name,
                cuda_device=2,
                # use_cuda=False
            )
        else:
            # c_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
            c_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
            q_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
            self.model = RetrievalModel(
                model_type=self.model_type,
                context_encoder_name=c_encoder_name,
                query_encoder_name=q_encoder_name,
                cuda_device=2
            )

    def get_top_n(self, question, candidates, n=10, tau=50):
        self.model.prediction_passages = None  # need to clear it manually for new corpus

        titles = ['n/a' for _ in range(len(candidates))]
        gold_passages = ['n/a' for _ in range(len(candidates))]

        passages = [description for description in candidates]

        df = pd.DataFrame(data={'title': titles, 'passages': passages, 'gold_passage': gold_passages})

        passages, doc_scores, _, _, _ = self.model.predict([question], prediction_passages=df, retrieve_n_docs=n)

        top_candidates = []
        for i in range(len(passages[0])):
            if tau is None or doc_scores[0][i] > tau:  # thresholding
                for c in candidates:
                    if passages[0][i] == c:
                        top_candidates.append(c)

        return top_candidates


class IRSentenceBert(IRModel):  # Sentence Bert model
    def __init__(self, model_path: str = None):
        self.model_type = "sentence_bert"

        # 'sentence-transformers/all-mpnet-base-v2'
        # 'sentence-transformers/all-MiniLM-L6-v2'
        if model_path is None:
            model_path = 'sentence-transformers/all-mpnet-base-v2'
        self.model_path = model_path

        self.model = SentenceTransformer(self.model_path)
        self.model.to("cuda:4")
        print("haha")

    def get_top_n(self, question, candidates, n=10, tau=0.2):
        question_emb = self.model.encode(question)
        question_emb = torch.tensor(question_emb).unsqueeze(0)

        passages = [description for description in candidates]

        embeddings = self.model.encode(passages)
        embeddings = torch.tensor(embeddings).transpose(0, 1)

        scores = torch.mm(question_emb, embeddings).squeeze()
        sorted_scores, indices = torch.sort(scores, descending=True)

        try:
            indices = indices.tolist()[:n]
        except TypeError:
            indices = [indices]
            scores.unsqueeze_(dim=0)

        top_candidates = []
        for idx in indices:
            if tau is None or scores[idx] > tau:
                top_candidates.append(candidates[idx])

        return top_candidates



if __name__ == '__main__':
    # model0 = IRBm25()
    # model1 = IRRetrieval(model_name="out_makeup")
    model1 = IRRetrieval()
    # model2 = IRSentenceBert()
    # model2.get_top_n()