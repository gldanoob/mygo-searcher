# Semantic searcher for images from the anime *MyGO!!!*

A small tool to help MyGO fans communicate
- **Semantic search** is a search technique based on the matching the meaning of keywords instead of literal strings, which supports keywords with different wordings or in another language
    - Example: the image of Anon saying the line "是又怎樣？" can be found by querying "so what"

## Usage
- (Optional) Create and activate a virtual environment
`python3 -m venv venv`
`source venv/bin/activate`

- Install dependencies
`pip install -r requirements.txt`

## How it works
During dataset creation, the tool uses a pre-trained text embedding model to convert the set of captions into vectors in euclidean space. When a query is made, the query is converted into a vector, and consine similarity is used to find the k-most similar entries in the dataset.

Specifically, for embeddings in $\mathbb{R}^n$, the cosine similarity between two vectors $\vec{a}$ and $\vec{b}$ is defined as $$\frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}\text{.}$$

From the duality of the dot product in euclidean space, the cosine similarity is equivalent to the cosine of the angle between the $\vec{a}$ and $\vec{b}$. A property of the sentence embeddings is the "semantic direction" of the embedding space, which is illustrated by the well-known "King - Man + Women = Queen" analogy. Cosine similarity exploits this property by checking the angle between the query and the dataset entries.

Optimization of the search is done by packing the dataset into a feature matrix $\boldsymbol{X}$ and computing $X \cdot \vec{q}$, where $\vec{q}$ is the query vector. The entries are then normalized and sorted by their cosine similarity with the query.


## Custom data
