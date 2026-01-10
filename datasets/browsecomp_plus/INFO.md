BrowseComp-Plus (1K documents) (Chen et al., 2025). A multi-hop question-answering benchmark for DeepResearch (OpenAI, 2025) questions that requires reasoning over multiple different
documents. The benchmark provides a verified offline corpus of 100K documents that is guaranteed
to contain gold, evidence, and hard negative documents for each task. Following Sun et al. (2025),
we use 150 randomly sampled tasks as our evaluation set; we provide 1000 randomly chosen documents to the model or agent, in which the gold and evidence documents are guaranteed to exist.
We report the percentage of correct answers. The answer to each task requires piecing together information from several documents, making these tasks more complicated than S-NIAH despite also
requiring a constant number of documents to answer.
