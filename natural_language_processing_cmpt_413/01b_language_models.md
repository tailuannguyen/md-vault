# Language models

A language model estimates the probability of a phrase, sentence, paragraph, or document through conditional predictions. It can score text, complete a prefix, or generate new text.

## Vocabulary and the space of sentences

Tokenize a training corpus, then define a finite vocabulary $V$. A dictionary is one possible starting point; common corpus-based choices retain the most frequent words up to a size limit or words occurring at least a chosen number of times. Even a finite vocabulary generates infinitely many possible sentences because length is unbounded.

Counting complete sentences gives $P(s)=C(s)/N_{\mathrm{sentences}}$. This assigns zero probability to every unseen sentence, including reasonable combinations of familiar words. Language models instead reuse evidence about smaller units across sentences.

Words outside the vocabulary are **out of vocabulary**, or OOV. In a closed-vocabulary model, map them to `<UNK>`. To learn that token's probability:

1. Count words in training data and choose the vocabulary using only those data.
2. Replace rare or excluded training words with `<UNK>`.
3. Train on the transformed corpus; transform validation and test text with the same vocabulary and replacement rule.

Adding a token name alone does not teach its probability. A uniform fallback is possible, but must be part of a normalized distribution rather than extra mass appended to existing predictions. `<UNK>` handles unseen word identities; smoothing handles unseen combinations of modeled tokens.

## Chain rule and n-gram approximation

The chain rule for a fixed-length sequence is exact:

$$
P(w_{1:T})=\prod_{t=1}^{T}P(w_t\mid w_{1:t-1}).
$$

For “the cat sat,” this gives $P(\text{the})P(\text{cat}\mid\text{the})P(\text{sat}\mid\text{the cat})$. The difficulty is estimating distributions conditioned on arbitrarily long histories.

An **$n$-gram model** makes a Markov assumption: only the previous $n-1$ tokens matter for the next prediction.

$$
P(w_t\mid w_{1:t-1})\approx P(w_t\mid w_{t-n+1:t-1}).
$$

Unigrams use no history, bigrams use one preceding token, and trigrams use two. Increasing $n$ retains more local information but raises the number of possible combinations to roughly $|V|^n$, many of which never occur in training. Longer context trades statistical reliability for modeling capacity.

Neural language models retain conditional sequence prediction but learn parameterized functions rather than separate counts for every n-gram. Other extensions can incorporate syntax, topics, or downstream objectives.

## Maximum likelihood estimation

Assuming independent training sentences $s^{(1)},\ldots,s^{(m)}$, maximize their likelihood:

$$
L(\theta)=\prod_{j=1}^{m}P(s^{(j)};\theta),\qquad
\log L(\theta)=\sum_{j=1}^{m}\log P(s^{(j)};\theta).
$$

Logarithms preserve the maximizer, turn products into sums, and reduce numerical underflow. For history $h$, define $C(h,w)$ as the number of times $w$ follows $h$, and $C(h)=\sum_wC(h,w)$ as its total number of prediction events. Maximum likelihood gives:

$$
\hat P(w\mid h)=\frac{C(h,w)}{C(h)},\qquad C(h)>0.
$$

For unigrams, $\hat P(w)=C(w)/N$, where $N$ counts all modeled token events. Each history's probabilities sum to one. Unseen successors receive zero probability, and an entirely unseen history has no defined count ratio.

The [[natural_language_processing_cmpt_413/language_model_derivations#Maximum likelihood for a categorical distribution|categorical and unigram proof]] and [[natural_language_processing_cmpt_413/language_model_derivations#Conditional maximum likelihood for bigrams|conditional bigram proof]] derive these ratios with Lagrange multipliers and explain zero-count cases.

## Smoothing

Smoothing reallocates mass so rare and unseen events receive probability. An unseen n-gram can otherwise make an entire sentence's probability zero even when its words are known. Distribution shift increases this risk: news and Shakespeare use different combinations.

### Additive smoothing

Let $\mathcal W$ be the prediction vocabulary, including `<UNK>` and STOP when modeled. Add-alpha smoothing adds the same pseudo-count to every candidate:

$$
P_\alpha(w\mid h)=\frac{C(h,w)+\alpha}{C(h)+\alpha|\mathcal W|},\qquad\alpha>0.
$$

Add-one, or Laplace smoothing, sets $\alpha=1$. Counts $(3,1,0)$ become probabilities $(4/7,2/7,1/7)$. The denominator adds a pseudo-count for **every** candidate, preserving normalization. An unseen history becomes uniform. Add-one often allocates too much mass to unseen events in large vocabularies; smaller $\alpha$ gives gentler smoothing.

### Linear and recursive interpolation

Jelinek–Mercer interpolation mixes context lengths. A trigram mixture can include a uniform component:

$$
P_{\mathrm{mix}}(w\mid u,v)=
\lambda_3\hat P(w\mid u,v)+\lambda_2\hat P(w\mid v)
+\lambda_1\hat P(w)+\lambda_0\frac1{|\mathcal W|},
$$

with $\lambda_k\geq0$ and $\sum_{k=0}^{3}\lambda_k=1$. A mixture of normalized distributions is normalized; a positive uniform weight ensures positive probability for every candidate. If a history is unseen, omit its undefined MLE component and redistribute its weight, or use an explicitly defined fallback.

Interpolation can also be recursive. Let $P_r$ use a history of length $r-1$, and let $h'$ drop the oldest token in $h$:

$$
P_r(w\mid h)=\lambda_r(h)\hat P(w\mid h)
+[1-\lambda_r(h)]P_{r-1}(w\mid h'),
\qquad P_0(w)=\frac1{|\mathcal W|}.
$$

At the unigram level the lower-order distribution is uniform; at higher levels it is a smoother shorter-context model. For an unseen history, set $\lambda_r(h)=0$ and use the lower-order distribution directly. Weights can be global or history-dependent, allowing reliable histories more weight on specific estimates. Tune weights and smoothing strength on held-out validation text.

## Boundaries, stopping, and generation

Start markers supply missing initial context; they are conditioning symbols, not generated words. For a trigram model, set $w_{-1}=w_0=\texttt{<s>}$ and append $w_{T+1}=\mathrm{STOP}$:

$$
P(s)=\prod_{t=1}^{T+1}P(w_t\mid w_{t-2},w_{t-1}).
$$

For “the cat,” this includes $P(\text{the}\mid\texttt{<s>},\texttt{<s>})$, $P(\text{cat}\mid\texttt{<s>},\text{the})$, and $P(\mathrm{STOP}\mid\text{the},\text{cat})$. The stopping factor distinguishes a complete sentence from a prefix.

Generate by initializing boundary context, sampling a token, updating the history, and repeating until STOP. Completion starts this procedure from an existing prefix.

Without STOP, unigram probabilities $P(a)=P(b)=1/2$ give total mass one at **each** fixed length. Summing over all lengths diverges. Locally normalized predictions alone do not define a distribution over finite sentences.

With unigram STOP probability $q>0$, total probability for sentences containing $n$ non-STOP tokens is $q(1-q)^n$. Summing over $n=0,1,\ldots$ gives one, including the empty sentence. The [[natural_language_processing_cmpt_413/language_model_derivations#Normalization with a STOP symbol|STOP normalization proof]] shows the intermediate steps. Context-dependent models also need eventual termination with probability one; merely making STOP available is insufficient.

## Evaluation and data splits

**Extrinsic evaluation** places the model in a downstream system such as translation, speech recognition, or question answering and measures system performance. It tests practical usefulness but can be expensive and gives indirect feedback about the language model.

**Intrinsic evaluation** scores unseen text directly under the model. It is faster, but better likelihood need not produce a better downstream system.

Train parameters, vocabulary, and preprocessing statistics on training data. Use validation data for smoothing, learning rates, penalties, thresholds, or checkpoint selection. Reserve test data for evaluation after those decisions. Repeated selection using test scores leaks information into the chosen system even without gradient training on test examples.

## Held-out loss and perplexity

For a test corpus with $M$ scored token events and observed-token conditional probabilities $p_t$:

$$
P(\mathcal T)=\prod_{t=1}^{M}p_t,\qquad
\ell=\frac1M\log_2P(\mathcal T)=\frac1M\sum_{t=1}^{M}\log_2p_t.
$$

Raw corpus probability shrinks as more events are included. Normalizing the log-probability by token count compares average predictive quality. Empirical cross-entropy in bits and perplexity are:

$$
H=-\ell,\qquad\mathrm{PPL}=2^H=P(\mathcal T)^{-1/M}
=\left(\prod_{t=1}^{M}\frac1{p_t}\right)^{1/M}.
$$

Perplexity is the inverse geometric mean probability of observed events. It measures predictive uncertainty, often interpreted as an effective branching factor. Uniform predictions over $K$ candidates give $H=\log_2K$ and $\mathrm{PPL}=K$. Probabilities $(1/2,1/4)$ give $H=1.5$ bits and perplexity $\sqrt8\approx2.828$.

Higher mean log-probability and lower perplexity indicate better prediction on the **same** held-out events. An observed zero probability gives infinite perplexity; rare, very low-probability events can dominate loss.

Compare on the same corpus, tokenization, vocabulary/OOV policy, and boundary convention, including whether STOP counts in $M$. Different domains or token definitions change the task, so their perplexities are not directly comparable. Perplexity evaluates sequence prediction, not every aspect of grammaticality, usefulness, or downstream accuracy.

## Advanced smoothing

Discounting removes mass from observed counts and reallocates it to unseen events. Backoff uses a lower-order model where higher-order evidence is insufficient, with weights preserving normalization. Kneser–Ney uses continuation counts—how many distinct contexts a word appears in—for its lower-order distribution rather than relying only on token frequency.

These methods require careful mass accounting. Web-scale “stupid backoff” returns unnormalized scores, which cannot directly serve as probabilities for perplexity.
