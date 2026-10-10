# Language models

A language model assigns sequence probabilities through conditional prediction. Count-based models need smoothing, stopping conventions, and held-out evaluation.

## Sequence probability and n-grams

Tokenization defines the events being predicted. Build a vocabulary from training data and apply a consistent unknown-word policy, commonly `<UNK>`.

The chain rule is exact:

$$P(w_{1:T})=\prod_{t=1}^{T}P(w_t\mid w_{1:t-1}).$$

An $n$-gram model approximates each history by the previous $n-1$ tokens. Unigrams use no history; bigrams use one token; trigrams use two. Longer histories retain more information but create more sparse combinations.

Maximum likelihood chooses parameters that make the observed training data most probable. For an observed history $h$:

$$\hat P(w\mid h)=\frac{C(h,w)}{C(h)}.$$

This shares local counts across sentences but gives unseen successors zero probability. An unseen history needs a fallback.

Neural language models retain the same probability factorization but parameterize the conditional distributions differently.

## Smoothing

Add-alpha smoothing distributes pseudo-counts across the prediction vocabulary:

$$P_\alpha(w\mid h)=\frac{C(h,w)+\alpha}{C(h)+\alpha\lvert V\rvert}.$$

For counts $(3,1,0)$, add-one gives $(4/7,2/7,1/7)$. The denominator must include the pseudo-count for every candidate.

Interpolation combines normalized unigram, bigram, and trigram distributions with nonnegative weights summing to 1. Tune these weights and smoothing strength on validation text.

## Boundaries and generation

Start markers supply initial context. STOP is a predicted event included in a complete utterance's probability. Generate by repeatedly sampling a token, updating the history, and stopping at STOP.

Without termination, fixed-length distributions do not automatically form a normalized distribution over all finite strings. For unigram STOP probability $q>0$, total mass for length $n$ is $(1-q)^nq$; summing from $n=0$ gives 1. This includes the empty string. Context-dependent models also need termination with probability 1.

## Evaluation and data splits

Train parameters and preprocessing statistics on training data. Use validation data for smoothing, learning rates, penalties, thresholds, or checkpoint selection. Reserve test data for evaluation after those decisions.

Repeated model selection using test scores leaks information into the chosen system even without gradient training on the test examples.

## Held-out loss and perplexity

For $M$ scored token events with observed-token probabilities $p_t$:

$$H=-\frac1M\sum_t\log_2p_t,\qquad \mathrm{PPL}=2^H.$$

Lower perplexity means higher average probability on the held-out text. Two probabilities $(1/2,1/4)$ give $H=1.5$ bits and perplexity $\sqrt8\approx2.828$.

Compare on the same corpus, tokenization, vocabulary/OOV policy, and boundary convention. A zero probability gives infinite perplexity. Better intrinsic likelihood does not guarantee better downstream performance.

## Advanced smoothing

Optional advanced smoothing methods include discounting, Kneser-Ney continuation counts, normalized backoff, and an unnormalized “stupid backoff” score. That score is not directly suitable for probabilistic perplexity.
