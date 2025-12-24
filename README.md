# NLP Semantic Clustering: Amazon Reviews

A comparison of traditional vs. modern vector representation techniques for unsupervised clustering of textual data.

## Features
- **Data Preprocessing**: Conversion of raw text into numerical vectors.
- **Techniques Compared**:
    - **Bag-of-Words (BoW)**: Traditional frequency-based approach.
    - **Sentence Transformers**: Modern neural embeddings using the `thenlper/gte-small` model.
- **Clustering**: Implementation of the K-Means algorithm.
- **Visualization**: Dimensionality reduction via PCA (Principal Component Analysis) to visualize clusters in 2D.

## Evaluation
The project uses the **Purity Score** to evaluate how well the unsupervised clusters align with ground-truth sentiment labels.

## Tech Stack
- Python, Scikit-learn, Sentence-Transformers, Pandas, Matplotlib.

## Installation
`pip install sentence-transformers scikit-learn pandas`
