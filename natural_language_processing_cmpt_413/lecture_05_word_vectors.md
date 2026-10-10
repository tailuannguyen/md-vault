# Word vectors

Word vectors share information across vocabulary items. They can be learned from context counts or prediction tasks and combined into neural document and language models.

## Word representations

### From identity to meaning

One-hot vectors encode vocabulary identity, but different words have zero cosine similarity. Dense embeddings use a smaller shared space, allowing words with similar usage to share information.

The distributional hypothesis says words in similar contexts tend to have similar meanings. It motivates both count-based vectors and predictive embedding methods. Similar contexts can also surround antonyms, so distributional similarity is not identical to synonymy.

### Lookup and adaptation

An embedding table $E\in\mathbb R^{\lvert V\rvert\times d}$ stores one row per word. Lookup selects that row; training can update it through downstream gradients.

Choose among random initialization with training, pretrained fixed vectors, or pretrained vectors with fine-tuning. Their usefulness depends on data coverage and the downstream task.

**Static** vectors are fixed for each word type. **Contextual** vectors vary with surrounding tokens, allowing “river bank” and “bank account” to receive different representations.

### Cosine similarity

For nonzero vectors:

$$\cos(u,v)=\frac{u^\top v}{\lVert u\rVert\lVert v\rVert}.$$

Cosine compares direction rather than magnitude.

## Count-based word vectors

### Count the contexts

Build a matrix $F$ with target words as rows and context types as columns. Entry $F_{ij}$ counts occurrences of word $i$ in context $j$. A row represents a word's usage; in a term-document matrix, columns represent documents.

Context choices matter: document membership, nearby words, window width, and direction can produce different representations. Target and context vocabularies need not be identical.

### Weight the evidence

Raw counts overemphasize common contexts. TF-IDF weights document terms by their frequency and distinctiveness:

$$\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\log\frac{D}{df_t}.$$

Here $df_t$ counts documents containing the term. Log-scaled term frequency and smoothed IDF variants are possible; fit statistics on training data.

PMI measures association relative to independence:

$$\operatorname{PMI}(w,c)=\log_2\frac{P(w,c)}{P(w)P(c)},\qquad
\operatorname{PPMI}(w,c)=\max(\operatorname{PMI}(w,c),0).$$

Compute the joint and marginals from the same matrix total. For pair count 6, row total 11, column total 7, and overall total 19:

$$\operatorname{PPMI}=\log_2\frac{6\cdot19}{11\cdot7}\approx0.566.$$

Rare pairs can have exaggerated PMI. Additive smoothing or a flattened context marginal can reduce this effect. After smoothing counts, recompute the total and marginals.

### Compress with SVD

Factor the weighted matrix as $X=U\Sigma V^\top$ and keep its largest $k$ singular components. Word embeddings are rows of $U_k\Sigma_k^\alpha$, with the exponent defining a scaling choice.

Truncated SVD creates a low-rank approximation and shares correlated context structure. Smaller vectors are convenient downstream, but compression can lose distinctions. Singular values of an uncentered matrix are not automatically PCA variances.

## Neural models for text

### Neural bag of words

For a nonempty document, average token embeddings and pass the result through hidden layers and a class softmax:

$$a=\frac1T\sum_{i=1}^{T}e_{w_i},\qquad h=f(Wa+b),\qquad
p=\operatorname{softmax}(Uh+b_o).$$

A deep averaging network adds more hidden transformations. Embeddings may be fixed or trained jointly with the classifier. The representation is order-insensitive, even when the layers after pooling are nonlinear.

### Feedforward neural language model

Concatenate a fixed window of context embeddings instead of averaging:

$$x=[e_{w_{t-n}};\ldots;e_{w_{t-1}}],\qquad
h=f(Wx+b),\qquad P(w_t\mid\mathrm{context})=\operatorname{softmax}(Uh+b_o).$$

Here $n$ is the number of context tokens, whereas an $n$-gram model uses $n-1$ context tokens. Concatenation retains the position of each vector within the window.

Train to minimize $-\log P(w_t\mid\mathrm{context})$. Observed next tokens provide supervision without manual annotation. The embeddings and network are shared across examples, although input weights can differ by window position. Increasing window size increases those parameters.

### Beyond a fixed window

An RNN updates a hidden state using previous state and the current token, reusing its parameters over time. ELMo's language-model states and transformer encoders provide contextual representations.

Autoregressive prediction must use only preceding tokens. Other contextual objectives may permit surrounding context, so the conditioning rule must match the training task.

### Common training recipe

The output is a classification distribution: document classes for neural BOW, vocabulary tokens for an LM. Use softmax and cross-entropy, differentiate with backpropagation, and assess generalization with held-out evaluation.
