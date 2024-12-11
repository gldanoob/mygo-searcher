import asyncio
import json
import os
from typing import Literal

import numpy as np
import openai
from dotenv import load_dotenv
from utils import check_literal

load_dotenv()
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_KEY"))


class Dataset():
    def __init__(self) -> None:
        self.sentences = []
        self.embeddings = np.zeros((0, 3072))

    def get_entry(self, index: int) -> tuple[str, np.ndarray]:
        return self.sentences[index], self.embeddings[index]

    def get_all_entries(self) -> tuple[list[str], np.ndarray]:
        return self.sentences, self.embeddings

    def add_entries(self, sentences: list[str], embeddings: np.ndarray) -> None:
        if len(sentences) != embeddings.shape[0]:
            raise ValueError("Mismatch in embeddings and sentences")

        self.sentences.extend(sentences)
        self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)


class SemanticSearcher():
    def __init__(self, model='text-embedding-3-large') -> None:
        # load sqlite db
        self.dataset = Dataset()
        self.model = model

    async def add_sentences(self, sentences: list[str]) -> None:
        embeddings = await self._get_embeddings(sentences)

        # save embeddings
        # np.save("data/embeddings.npy", embeddings)

        self.dataset.add_entries(sentences, embeddings)

    async def search(self, target: str, k=5) -> list[str]:
        indices, target_embedding = await self._k_most_similar(target, k)

        # get sentences
        sentences = []
        for index in indices:
            sentence, _ = self.dataset.get_entry(index)
            sentences.append(sentence)

        return sentences

    async def _k_most_similar(self, target: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k < 1:
            raise ValueError("Invalid k value")

        sentences, embeddings = self.dataset.get_all_entries()

        # embed target
        target_embedding = await self._get_embeddings([target])

        if target_embedding.shape[0] != 1 and target_embedding.shape[1] != embeddings.shape[1]:
            raise ValueError("Invalid target embedding shape")

        # cosine similarity
        similarity: np.ndarray = target_embedding @ embeddings.T / \
            (np.linalg.norm(target_embedding) * np.linalg.norm(embeddings, axis=1))

        # sort by similarity and index
        indices = np.argsort(similarity.flatten())[::-1]

        return indices[:k], target_embedding

    async def _get_embeddings(self, sentences: list[str]) -> np.ndarray:
        res = await client.embeddings.create(input=sentences, model=self.model)
        embeddings = np.array([item.embedding for item in res.data])

        if embeddings.shape[0] != len(sentences):
            raise ValueError("Mismatch in embeddings and sentences")

        return embeddings
