import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from transformers import AutoTokenizer, AutoModel
from langchain_openai import OpenAIEmbeddings

from embodied_cd.common.print_utils import *


class TfidfPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def __call__(self, sentence1: str, sentence2: str) -> float:
        if isinstance(sentence1, str):
            return self.pipe([sentence1, sentence2])
        else: # if list
            score = 0
            for sen1, sen2 in zip(sentence1, sentence2):
                score += self.pipe([sen1, sen2])
            return score
    
    def pipe(self, sentences: list) -> np.array:
        if any(not re.search(r"\w", s or "") for s in sentences):
            return 0.0  # empty / stop-word-only text has no tf-idf vocabulary
        try:
            matrix = self.vectorizer.fit_transform(sentences)
        except ValueError:  # "empty vocabulary" (all tokens are stop words)
            return 0.0
        cossim = linear_kernel(matrix, matrix)
        return cossim[0][1]

class SentenceSimilarityPipeline:
    def __init__(self, model='sentence-transformers/paraphrase-MiniLM-L6-v2', tokenizer=None):
        if not isinstance(model, str) or 'sentence-transformers' in model: 
            self.pipe = SentenceTransformerPipe(model, tokenizer)
        else:
            self.pipe = OpenAIEmbeddingPipe(model)
    
    def __call__(self, sentence1: str, sentence2: str) -> float:
        if isinstance(sentence1, str):
            return self.pipe(sentence1, sentence2)
        else: # if list
            score = 0
            for sen1, sen2 in zip(sentence1, sentence2):
                score += self.pipe(sen1, sen2)
            return score


class SentenceTransformerPipe:
    def __init__(self, model: str='sentence-transformers/paraphrase-MiniLM-L6-v2', tokenizer=None):
        if isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model, device_map='auto')
            print_warn("[Pipe: SentenceTransformer] Use non-fine-tuned reward model!")
        else:
            print_warn("[Pipe: SentenceTransformer] Use fine-tuned reward model!")
            self.model, self.tokenizer = model, tokenizer
        self.device = self.model.device

    def __call__(self, sentence1: str, sentence2: str) -> float:
        embeddings = self.encode([sentence1, sentence2])
        cossim = F.cosine_similarity(embeddings[0], embeddings[1], dim=0) 
        return cossim.item()

    def encode(self, sentences: list) -> torch.Tensor:
        with torch.no_grad():
            input_ids = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**input_ids)
        sentence_embeddings = self.mean_pooling(model_output, input_ids['attention_mask'])
        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class OpenAIEmbeddingPipe:
    def __init__(self, model: str='text-embedding-3-large'):
        self.model = OpenAIEmbeddings(model=model)

    def __call__(self, sentence1: str, sentence2: str) -> float:
        embedding1 = self.model.embed_query(sentence1)
        embedding2 = self.model.embed_query(sentence2)
        cossim = cosine_similarity([embedding1], [embedding2])
        return cossim[0][0]
        
