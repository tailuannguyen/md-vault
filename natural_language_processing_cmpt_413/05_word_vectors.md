# Word vectors

Word vectors share information across vocabulary items. They can be learned from context counts or prediction tasks and combined into neural document and language models.

## Word representations

### From identity to meaning

One-hot vectors encode vocabulary identity, but different words have zero cosine similarity. Dense embeddings use a smaller shared space, allowing words with similar usage to share information.

The distributional hypothesis says words in similar contexts tend to have similar meanings. It motivates both count-based vectors and predictive embedding methods. Similar contexts can also surround antonyms, so distributional similarity is not identical to synonymy.

### Lookup and adaptation

An embedding table $E\in\mathbb R^{\lvert V\rvert\times d}$ stores one row per word. If $o_w$ is the word's one-hot column vector, lookup is mathematically $e_w=E^\top o_w\in\mathbb R^d$. Implementations select the row directly instead of multiplying through a mostly zero vector. Training can update the selected embedding through downstream gradients; repeated uses of a word accumulate contributions to the same parameters.

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

Here $D$ is the number of documents and $df_t$ counts documents containing the term, not total occurrences. One common log-scaled term frequency is

$$\operatorname{tf}(t,d)=
\begin{cases}1+\log_{10}\operatorname{count}(t,d),&\operatorname{count}(t,d)>0,\\
0,&\operatorname{count}(t,d)=0.
\end{cases}$$

Terms present in nearly every document have little IDF weight; a term present in all documents has zero weight under the unsmoothed formula. Smoothed IDF variants are possible. TF-IDF is useful for document similarity, query-document retrieval, and classifier features; it is a baseline weighting method rather than a typical modern dense embedding method. Fit statistics on training data when evaluating a learned classifier.

PMI measures association relative to independence:

$$\operatorname{PMI}(w,c)=\log_2\frac{P(w,c)}{P(w)P(c)},\qquad
\operatorname{PPMI}(w,c)=\max(\operatorname{PMI}(w,c),0).$$

Compute the joint and marginals from the same matrix total $N=\sum_{ij}F_{ij}$:

$$P(w_i,c_j)=\frac{F_{ij}}N,\qquad
P(w_i)=\frac{\sum_jF_{ij}}N,\qquad
P(c_j)=\frac{\sum_iF_{ij}}N.$$

PMI is zero at independence, positive for a pair more frequent than independence predicts, and negative for a less frequent pair. For zero observed joint probability and positive marginals, PMI tends to $-\infty$ and PPMI is set to zero. PPMI therefore discards negative association, rather than encoding it as a negative coordinate.

For pair count 6, row total 11, column total 7, and overall total 19:

$$\operatorname{PPMI}=\log_2\frac{6\cdot19}{11\cdot7}\approx0.566.$$

Rare pairs can have exaggerated PMI: one occurrence can look unusually informative when the marginals are tiny. Two approaches adjust that behavior.

**Weighted PMI** replaces the context marginal in the denominator with a normalized power distribution:

$$P_\alpha(c)=\frac{\operatorname{count}(c)^\alpha}
{\sum_{c'}\operatorname{count}(c')^\alpha},\qquad
\operatorname{PPMI}_\alpha(w,c)=\max\left(
\log_2\frac{P(w,c)}{P(w)P_\alpha(c)},0\right).$$

For $0<\alpha<1$, such as $\alpha=.75$, the distribution becomes flatter. With original context probabilities $.99$ and $.01$, normalized $.75$ powers give approximately $.9691$ and $.0309$. Increasing the denominator for the rare context reduces its PMI. This changes the context baseline; it does not add occurrences to the pair table.

**Additive smoothing** adds $\lambda>0$ to every cell of the chosen $W\times C$ matrix:

$$F'_{ij}=F_{ij}+\lambda,\quad N'=N+\lambda WC,\quad
F'_{i\cdot}=F_{i\cdot}+\lambda C,\quad
F'_{\cdot j}=F_{\cdot j}+\lambda W.$$

Recompute probabilities and PPMI from those adjusted counts. The smoothing dimensions matter: define the word and context sets first.

For example, use four words and five contexts:

| Word | computer | data | pinch | result | sugar |
|---|---|---|---|---|---|
| apricot | 0 | 0 | 1 | 0 | 1 |
| pineapple | 0 | 0 | 1 | 0 | 1 |
| digital | 2 | 1 | 0 | 1 | 0 |
| information | 1 | 6 | 0 | 4 | 0 |

The information/data cell has the unsmoothed value above. With add-2 smoothing, its count becomes 8, row total 21, column total 15, and overall total $19+2(4)(5)=59$:

$$\operatorname{PPMI}'(\mathrm{information},\mathrm{data})
=\max\left(\log_2\frac{8\cdot59}{21\cdot15},0\right)
\approx .583.$$

The same smoothing reduces apricot/pinch from $\log_2(19/4)\approx2.248$ to $\log_2(177/120)\approx.561$. Smoothing reduces extreme rare associations, but need not lower every entry.

### Compress with SVD

For a weighted word-context matrix $X\in\mathbb R^{W\times C}$ of rank $r$, the reduced singular value decomposition is

$$X=U\Sigma V^\top,\qquad
U\in\mathbb R^{W\times r},\quad \Sigma\in\mathbb R^{r\times r},\quad
V^\top\in\mathbb R^{r\times C}.$$

$U$ and $V$ have orthonormal columns, and $\Sigma$ is diagonal with singular values $\sigma_1\geq\cdots\geq\sigma_r>0$. Retain the largest $k\leq r$ components:

$$X_k=U_k\Sigma_kV_k^\top.$$

The shapes become $W\times k$, $k\times k$, and $k\times C$. Word embeddings are rows of $U_k\Sigma_k^\gamma$. Common choices are $\gamma=0$ (unscaled coordinates), $\gamma=1$ (singular-value weighting), and $\gamma=.5$ (intermediate weighting). This exponent is separate from the context-smoothing exponent above.

Truncated SVD creates a low-rank approximation and shares correlated context structure. If two words occur with different but correlated contexts, compression can bring their representations closer even without identical count rows. Smaller vectors are convenient downstream, but compression can lose distinctions. Singular values of an uncentered matrix are not automatically PCA variances.

Conventional full dense SVD of a square vocabulary matrix can cost $O(|V|^3)$; sparse or truncated methods change that tradeoff. The count/SVD pipeline fits global matrix structure separately from a downstream task. Its objective is not the task's classification loss. SVD-derived vectors can still initialize a trainable embedding table; that later fine-tuning is distinct from recomputing SVD jointly with the classifier.

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

Here $n$ is the number of context tokens, whereas an $n$-gram model uses $n-1$ context tokens. Concatenation retains the position of each vector within the window. If embeddings have dimension $d$ and the hidden layer has width $H$, then

$$x\in\mathbb R^{nd},\quad W\in\mathbb R^{H\times nd},\quad
b,h\in\mathbb R^H,\quad U\in\mathbb R^{|V|\times H},\quad
b_o\in\mathbb R^{|V|}.$$

For predicting “mat” after “the cat sat on the,” five embeddings make a $5d$-dimensional input, and the output has one score per vocabulary word. “The” uses the same embedding row at both positions, but the input weight blocks for those positions can differ.

Train to minimize $-\log P(w_t\mid\mathrm{context})$. Observed next tokens provide supervision without manual annotation. The embeddings and network are shared across examples, although input weights can differ by window position. Increasing window size increases those parameters.

### Beyond a fixed window

An RNN updates a hidden state using previous state and the current token:

$$h_t=f(h_{t-1},e_{w_t}),\qquad
P(w_t\mid w_{<t})=\operatorname{softmax}(Uh_{t-1}+b_o).$$

The state $h_{t-1}$ summarizes the preceding tokens before predicting $w_t$. The same transition parameters are reused across positions, avoiding a separate input weight block for every position in a larger fixed window. A hidden state is a learned compression of history, not a guarantee that every earlier detail is retained.

ELMo trains forward and backward stacked LSTM language models and uses their token-specific hidden states for contextual representations. Unlike a static lookup, these states change with the sentence. Transformers also transform input embeddings into contextual states; the permitted attention context depends on the task.

Autoregressive prediction must use only preceding tokens. Other contextual objectives may permit surrounding context, so the conditioning rule must match the training task.

### Common training recipe

The output is a classification distribution: document classes for neural BOW, vocabulary tokens for an LM. Use softmax and cross-entropy, differentiate with backpropagation, and assess generalization with held-out evaluation.
