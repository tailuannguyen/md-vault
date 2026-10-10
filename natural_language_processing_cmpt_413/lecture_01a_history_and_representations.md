# NLP history and representations

The core modeling choices are representation, composition, and prediction. NLP has progressed from hand-written rules to statistical models and learned representations.

## From rules to learned representations

Rule-based systems encode human-designed patterns. Statistical systems estimate parameters from corpora, often using hand-designed features. Neural systems learn intermediate representations as part of training. More data and compute helped drive this progression, but task definition and evaluation remain necessary.

Historical systems and RLHF illustrate this progression at a high level.

## Representing and composing language

One-hot vectors identify words but do not encode similarity: two distinct words are orthogonal. Dense embeddings place words in a shared space where similar usage can support similar representations.

Averaging word vectors gives a simple sentence representation, but discards order. “dog bites person” and “person bites dog” have the same average. Sequence models can retain order-dependent information.
