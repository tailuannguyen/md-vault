# Natural language processing

CMPT 413/713

## Course overview

Natural language processing examines the theoretical and applied problems of building systems that extract and represent meaning from sentences and whole discourses. It draws on linguistics, cognitive psychology, artificial intelligence, and computing science.

Human languages differ from programming languages: their structure and meaning are harder to encode explicitly, and systems must learn from observations while handling ambiguity. NLP uses computer algorithms to analyze languages such as English, Korean, and thousands of others.

Applications include translating speech between languages during a phone call, animating a person speaking written text, interpreting instructions for a robot, discovering protein–drug interactions in medical abstracts, and analyzing ancient scripts. These applications connect language understanding and representation to practical tasks.

The course introduces basic linguistic concepts and algorithms for processing text with probabilistic models and neural networks. It requires a strong background in probability, linear algebra, and calculus, along with confidence implementing algorithms. Coding assignments use Python and PyTorch. Familiarity with machine learning and deep learning is recommended.

## Course topics

- Language models.
- Word representations.
- Supervised machine learning for NLP.
- Neural models for NLP.
- Sequence labeling.
- Machine translation.
- Semantics.
- NLP applications.

## Educational goals

- Understand the algorithms and techniques used in NLP, including how computers represent words and model text as sequences.
- Understand the mathematics and training of common models, including language models, recurrent neural networks, and transformers.
- Understand basic NLP tasks such as part-of-speech tagging and how models support applications such as machine translation.

## Note outline

- [01: NLP introduction and course setup](natural_language_processing_cmpt_413/01_nlp_introduction.md): language ambiguity, NLP tasks, and course expectations.
- [01a: NLP history and representations](natural_language_processing_cmpt_413/01a_history_and_representations.md): rules and learned representations, scale and sparsity, the Vauquois triangle, context learning, and composition.
- [01b: Language models](natural_language_processing_cmpt_413/01b_language_models.md): sequence probabilities, vocabulary handling, n-grams, smoothing and interpolation, generation, evaluation, and perplexity.
- [Language model derivations](natural_language_processing_cmpt_413/language_model_derivations.md): the companion to note 01b, with maximum-likelihood estimation proofs and normalization over variable-length strings using STOP.
- [02: Text classification and Naive Bayes](natural_language_processing_cmpt_413/02_text_classification.md): document models, Naive Bayes, event models, data splits, and classification metrics.
- [03: Logistic regression](natural_language_processing_cmpt_413/03_logistic_regression.md): binary and multiclass probabilities, loss gradients, regularization, and learned features.
- [Logistic regression derivations](natural_language_processing_cmpt_413/logistic_regression_derivations.md): the companion to note 03, with log odds, likelihood and cross-entropy, chain-rule gradients, and L1 sparsity.
- [04: Neural networks](natural_language_processing_cmpt_413/04_neural_networks.md): layers, activations, backpropagation, optimization, and generalization.
- [Neural network training derivations](natural_language_processing_cmpt_413/neural_network_training_derivations.md): the companion to note 04, with nonlinearity and XOR, computational graphs, worked backpropagation, and loss gradients.
- [05: Word vectors](natural_language_processing_cmpt_413/05_word_vectors.md): word representations, cosine similarity, co-occurrence counts, weighting, SVD, and neural models for text.
- [06: Word2vec](natural_language_processing_cmpt_413/06_word2vec.md): skip-gram, negative sampling, CBOW, GloVe, FastText, embedding evaluation, and sentence representations.
- [Word2vec training derivations](natural_language_processing_cmpt_413/word2vec_training_derivations.md): the companion to note 06, with full-softmax gradients, negative-sampling gradients, and the shifted-PMI derivation.
