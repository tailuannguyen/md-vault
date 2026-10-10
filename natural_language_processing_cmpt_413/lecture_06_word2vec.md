# Word2vec

Word2vec learns word vectors by predicting words from corpus context. Nearby words supply the training targets automatically, so no manually labeled examples are needed. The prediction problem is a **pretext task**: the learned vectors are useful beyond the classifier that produced them.

## Skip-gram: construct pairs and predict contexts

Start with a corpus $w_1,\ldots,w_T$, vocabulary $V$, embedding dimension $d$, and context-window half-width $m$. Skip-gram predicts each neighboring word from its center word. A window includes up to $m$ tokens on either side and excludes the center itself.

For the fragment “a tablespoon of apricot jam a pinch,” a window of two around **apricot** gives these positive pairs:

- `(apricot, tablespoon)`
- `(apricot, of)`
- `(apricot, jam)`
- `(apricot, a)`

Each word has a center vector $u_t$ and a context vector $v_c$, both in $\mathbb R^d$. These are separate parameter tables for the two roles, giving $2|V|d$ parameters. Their dot product scores a center-context pair:

$$P(c\mid t)=\frac{\exp(u_t^\top v_c)}{\sum_{k\in V}\exp(u_t^\top v_k)}.$$

This softmax is normalized over the context vocabulary. Raising one candidate's probability affects the others through the shared denominator.

### Objective and training

Let $\mathcal C_i$ be the valid context positions within the window around corpus position $i$. Minimize the negative log-likelihood:

$$J(\theta)=-\frac1T\sum_{i=1}^T\sum_{j\in\mathcal C_i}
\log P(w_j\mid w_i;\theta).$$

Here $i,j$ denote corpus positions; $t,c$ in the pair formulas denote word identities. Averaging over pairs instead of positions changes the constant scaling of the objective.

Initialize both tables randomly, generate pairs from the corpus, compute gradients for each pair or batch, and update the parameters with SGD. The objective is generally nonconvex when both vector tables are learned together.

For pair loss $\ell=-\log P(c\mid t)$, the gradients are

$$\nabla_{u_t}\ell=-v_c+\sum_{k\in V}P(k\mid t)v_k,$$

$$\nabla_{v_k}\ell=\bigl(P(k\mid t)-\mathbf1[k=c]\bigr)u_t.$$

The center gradient compares the observed context with the model's expected context. Every vocabulary context row generally receives a gradient, so full-softmax training costs roughly $O(|V|d)$ per pair. This motivates negative sampling.

The [[natural_language_processing_cmpt_413/word2vec_training_derivations#Full-softmax loss and gradients|full softmax derivation]] shows each differentiation step and how the gradients produce SGD updates.

## Skip-gram with negative sampling

SGNS changes the task to binary classification: distinguish observed word-context pairs from pairs constructed with noise words. For each positive pair, draw $K$ negative contexts from a noise distribution $Q$; a small number such as 5-20 avoids processing the full vocabulary.

For example, `(apricot, jam)` is an observed pair, while `(apricot, aardvark)` could be a noise pair. A sampled negative is a training label, not a claim that the two words can never occur together.

With $\sigma(s)=1/(1+e^{-s})$, the classifier predicts

$$P(y=1\mid t,c)=\sigma(u_t^\top v_c).$$

For positive $(t,c)$ and sampled noise words $n_1,\ldots,n_K$, minimize

$$L=-\log\sigma(u_t^\top v_c)-\sum_{i=1}^{K}\log\sigma(-u_t^\top v_{n_i}).$$

Training raises observed-pair scores and lowers sampled negative-pair scores. Only the center row and observed/sampled context rows are needed, giving roughly $O((K+1)d)$ work per positive pair. Repeated context rows must accumulate all their gradient contributions.

This sigmoid models an **observed-versus-noise label under the sampling setup**. It is not a normalized distribution over context words. SGNS therefore changes the learning objective rather than merely computing the original softmax more cheaply. The [[natural_language_processing_cmpt_413/word2vec_training_derivations#Negative-sampling gradients|negative-sampling derivation]] explains the binary-loss gradients and which rows are updated.

### Choose the noise distribution

A common choice uses corpus word counts $C(w)$:

$$Q(w)=\frac{C(w)^{3/4}}{\sum_{z\in V}C(z)^{3/4}}.$$

The exponent flattens the distribution, giving rare words more probability relative to frequent words. If two words have unigram probabilities 0.99 and 0.01, this transformation gives approximately 0.97 and 0.03. Frequent words still dominate, but the rarer word is sampled more often than under the original distribution.

### Relationship with PMI and count-based vectors

Under independently adjustable pair scores and noise equal to the empirical context unigram distribution, the ideal score is

$$u_t^\top v_c\approx\operatorname{PMI}_{\ln}(t,c)-\log K.$$

The [[natural_language_processing_cmpt_413/word2vec_training_derivations#Why SGNS gives shifted PMI|shifted-PMI derivation]] states the assumptions and optimizes a pair's expected objective. Using the $3/4$-power noise changes this expression, and finite embedding dimension couples scores that the idealized argument treats independently.

This connects predictive embeddings to word-context matrices. SGNS uses two unconstrained tables and a weighted logistic loss; truncated SVD uses orthonormal singular-vector factors and a squared reconstruction-error objective. SGNS is not universally equivalent to SVD of clipped PPMI.

## Continuous bag of words

CBOW reverses the prediction direction: aggregate surrounding context vectors and predict the center word. For a full window,

$$\bar v_i=\frac1{2m}\sum_{\substack{-m\le j\le m\\j\ne0}}v_{w_{i+j}},$$

$$P(w_i\mid\text{context})=
\frac{\exp(u_{w_i}^\top\bar v_i)}{\sum_{k\in V}\exp(u_k^\top\bar v_i)}.$$

At sentence boundaries, average over the available context tokens. Pooling discards their order. Each context window supplies one center-word target, whereas skip-gram supplies a separate prediction for each neighboring word.

CBOW is typically faster and often works well for frequent words. Skip-gram often works well for rarer words and smaller corpora. These are empirical tendencies; results also depend on the corpus and training settings.

## GloVe and FastText

### GloVe: fit global co-occurrence statistics

GloVe learns vectors from a global word-context count matrix $X$. Its objective is

$$J=\sum_{i,j:X_{ij}>0}f(X_{ij})
\bigl(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij}\bigr)^2.$$

Here $w_i$ and $\tilde w_j$ are word and context vectors, $b_i$ and $\tilde b_j$ are biases, and $X_{ij}$ counts their co-occurrence. The frequency-dependent weight $f$ lies between zero and one: it reduces the contribution of rare counts and caps the weight of frequent counts. The model fits log counts with weighted squared error.

A common final representation combines the two vectors for the **same word**, $w_i+\tilde w_i$. Learned neighborhoods can group “frog” with “frogs,” “toad,” and frog-related species names. Such examples illustrate the representation but are not sufficient evaluation by themselves.

### FastText: share character n-grams

FastText extends skip-gram by representing a word through shared character pieces, commonly n-grams of lengths 3-6. With boundary markers, the word `where` has character trigrams such as `<wh`, `whe`, `her`, `ere`, and `re>`.

If $G(w)$ is its set of character pieces, the pair score becomes

$$\sum_{g\in G(w)}u_g^\top v_c
=\left(\sum_{g\in G(w)}u_g\right)^\top v_c.$$

Sharing pieces helps related spellings and can construct vectors for unseen forms from known subwords. These are character n-grams, distinct from the word-token n-grams used in language models. The resulting word vector is still static: it does not change with the current sentence.

Pretrained vectors differ in training algorithm, corpus, dimension, vocabulary, and cased/uncased preprocessing. A method name alone does not determine their suitability.

## Evaluate embeddings

**Extrinsic evaluation** inserts embeddings into an actual NLP system, such as a sentiment classifier, and measures downstream performance. It is more expensive but directly tests usefulness. **Intrinsic evaluation** measures an intermediate property, such as word similarity or analogy retrieval; it is faster, but improvement need not transfer to the downstream task.

### Similarity and rank correlation

For nonzero vectors,

$$\cos(u,v)=\frac{u^\top v}{\|u\|\|v\|}.$$

Word-similarity datasets such as WordSim-353 provide human scores for word pairs. Compute a cosine score for each pair, rank the model scores and human scores, and compare their order with Spearman correlation:

$$\rho_s=\operatorname{corr}\bigl(\operatorname{rank}(s_{\text{model}}),
\operatorname{rank}(s_{\text{human}})\bigr).$$

This is Pearson correlation applied to ranks, using average ranks for ties. Different datasets emphasize different properties, including relatedness, similarity, and rare-word behavior.

### Analogies and experimental settings

For analogy $a:b::c:?$, form $q=e_b-e_a+e_c$ and retrieve

$$\hat d=\arg\max_{x\in V\setminus\{a,b,c\}}\cos(e_x,q).$$

Semantic examples include `Paris:France::Rome:Italy`; syntactic examples include `bad:worst::cool:coolest`. Offsets capture empirical patterns, not guaranteed algebraic identities. Report retrieval accuracy and distinguish semantic from syntactic performance.

Benchmark rankings can change with context-window size, corpus size, dimensionality, preprocessing, and evaluation procedure. A wider window is not uniformly better: settings that help one benchmark can hurt another. Interpret comparative tables together with their experimental conditions.

### Bias

Embeddings reflect corpus associations, including cultural stereotypes. For example, occupation words can associate differently with gender-related words. Such associations can change over time and correlate with occupational demographics; correlation does not establish causation.

Good similarity or analogy scores do not establish fair downstream behavior. A two-dimensional projection also shows only part of the representation.

## Sentence representations and extensions

### Average, weight, and remove a shared direction

A basic sentence representation averages its word vectors. A frequency-weighted version reduces the influence of common words:

$$s=\frac1T\sum_{i=1}^T\frac{a}{a+p(w_i)}e_{w_i},\qquad a>0,$$

where $p(w)$ is corpus word frequency. Rare words receive weights closer to one; frequent words receive smaller weights. This remains a bag-of-words representation.

To remove a common direction, collect sentence vectors as columns of a matrix and compute its leading left singular vector $r$, normalized to unit length. Remove each sentence vector's projection onto that direction:

$$s'=s-r(r^\top s).$$

The shared component may reflect common information, but it is not guaranteed to correspond specifically to syntax. Averaging and projection removal cannot recover discarded word order. Sequence models can represent order-dependent information.

### Beyond one static vector per word

- **Phrases and multi-word expressions:** represent units larger than a single token.
- **Sense embeddings:** learn multiple prototypes for a word. “Bank” can have financial and river senses; “jaguar” can refer to an animal or a car brand.
- **Dependency contexts:** use syntactic neighbors and relation labels rather than only nearby tokens. In “scientist discovers star,” the subject and object provide different contexts for “discovers.”
- **Hyperbolic representations:** use geometry suited to hierarchical branching, such as taxonomies, rather than an ordinary Euclidean space.
- **Cross-lingual embeddings:** place words from different languages in aligned spaces so translation equivalents can be close.
- **Cross-modal embeddings:** align language with other modalities, such as images of colored cars and their color words.
- **Contextual embeddings:** systems such as ELMo and BERT produce representations that depend on surrounding tokens, rather than selecting the same static vector for every occurrence.
