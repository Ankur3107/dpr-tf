# dpr-tf
Dense Passage Retrieval using tensorflow-keras on TPU

# Introduction

Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the defacto method. We can implement using dense representations, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework.

# Model Preparation

Bi-Encoder model using pre-trained base models i.e. bert-base-uncased.

![Bi-Model](https://ankur3107.github.io/assets/images/dpr/model-architechture.jpeg)

# In-Batch Training on TPU

![Bi-Model](https://ankur3107.github.io/assets/images/dpr/dpr-tpu-training-v2.jpeg)

# References

    DPR Paper: https://arxiv.org/pdf/2004.04906.pdf
    Blog: https://ankur3107.github.io/blogs/dense-passage-retriever/


Cited as:

@article{kumar2021dprtpu,
  title   = "The Illustrated Dense Passage Retreiver on TPU",
  author  = "Kumar, Ankur",
  journal = "ankur3107.github.io",
  year    = "2021",
  url     = "https://ankur3107.github.io/blogs/dense-passage-retriever/"
}