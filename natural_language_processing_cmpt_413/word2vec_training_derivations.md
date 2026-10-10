# Word2vec training derivations

This note develops the mathematics behind the training objectives in [[natural_language_processing_cmpt_413/06_word2vec|Lecture 6: Word2vec]]. It keeps the softmax gradients, negative-sampling gradients, and shifted-PMI argument together because they explain how the same word-context scores are learned under different objectives.

## Setup and notation

For vocabulary $V$ and embedding dimension $d$, each word has two independently learned vectors: a center vector $u_t\in\mathbb R^d$ and a context vector $v_c\in\mathbb R^d$. Here $t$ and $c$ are word identities, not positions in a sentence. The score for a pair is

$$s_{tc}=u_t^\top v_c.$$

We derive losses for one observed center-context pair. Corpus gradients sum or average the contributions from all pairs. All gradients below are for **loss minimization**; derivatives of the corresponding log-likelihood have the opposite sign.

## Full-softmax loss and gradients

### Rewrite the negative log probability

The softmax distribution and its normalizer are

$$P(k\mid t)=\frac{\exp(u_t^\top v_k)}{Z_t},\qquad
Z_t=\sum_{j\in V}\exp(u_t^\top v_j).$$

For observed context $c$, the loss is

$$\begin{aligned}
\ell(t,c)
&=-\log P(c\mid t)\\
&=-\log\left(\frac{\exp(u_t^\top v_c)}{Z_t}\right)\\
&=-u_t^\top v_c+\log Z_t.
\end{aligned}$$

This separates the observed pair's score from the normalization over every candidate context word.

### Differentiate with respect to the center vector

The first term gives $-v_c$. For the second, use the derivative of a logarithm and then the chain rule inside the sum:

$$\begin{aligned}
\nabla_{u_t}\ell
&=-v_c+\frac{1}{Z_t}\nabla_{u_t}Z_t\\
&=-v_c+\frac{1}{Z_t}\sum_{k\in V}
  \exp(u_t^\top v_k)\nabla_{u_t}(u_t^\top v_k)\\
&=-v_c+\sum_{k\in V}
  \frac{\exp(u_t^\top v_k)}{Z_t}v_k\\
&=-v_c+\sum_{k\in V}P(k\mid t)v_k.
\end{aligned}$$

The gradient is the model's expected context vector minus the observed context vector. Gradient descent therefore moves the center vector toward the observed context and away from the current probability-weighted average of alternatives.

### Differentiate with respect to each context vector

Consider any $v_k$. It affects the observed-score term only when $k=c$, but always appears in the normalizer:

$$\begin{aligned}
\nabla_{v_k}\ell
&=-\mathbf1[k=c]u_t
  +\frac{1}{Z_t}\nabla_{v_k}\sum_{j\in V}\exp(u_t^\top v_j)\\
&=-\mathbf1[k=c]u_t
  +\frac{\exp(u_t^\top v_k)}{Z_t}u_t\\
&=\bigl(P(k\mid t)-\mathbf1[k=c]\bigr)u_t.
\end{aligned}$$

For the observed context, the coefficient is $P(c\mid t)-1$. For every other candidate, it is $P(k\mid t)$. With finite scores, all candidates have positive softmax probability, so a single pair generally contributes gradients to **every context row**.

### Apply the updates

For learning rate $\eta>0$:

$$u_t\leftarrow u_t-\eta\left(-v_c+\sum_{k\in V}P(k\mid t)v_k\right),$$

$$v_k\leftarrow v_k-\eta\bigl(P(k\mid t)-\mathbf1[k=c]\bigr)u_t,
\qquad k\in V.$$

Compute all gradients from the same parameter values before applying updates. Updating $u_t$ first and then using its new value for the context derivatives would change the intended gradient step.

Each pair needs $|V|$ dot products of length $d$, giving roughly $O(|V|d)$ work. Learning both vector tables jointly is generally nonconvex because the scores depend bilinearly on their parameters.

## Negative-sampling gradients

The following derivation follows from the binary-classification loss. It explains which vector rows change when the full vocabulary softmax is replaced by sampled negative pairs.

### Derive the scalar binary-loss gradient

For label $y\in\{0,1\}$, score $s$, and $p=\sigma(s)$:

$$\ell_y(s)=-y\log p-(1-y)\log(1-p),\qquad
\sigma(s)=\frac{1}{1+e^{-s}}.$$

Since $dp/ds=p(1-p)$,

$$\begin{aligned}
\frac{d\ell_y}{ds}
&=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)\\
&=-y(1-p)+(1-y)p\\
&=p-y.
\end{aligned}$$

A positive pair has derivative $\sigma(s)-1$; a negative pair has derivative $\sigma(s)$. For $s=u_t^\top v_c$, the chain rule multiplies this scalar by $v_c$ for the center derivative and by $u_t$ for the context derivative.

### Combine one positive pair and its negative samples

For observed $(t,c)$ and noise words $n_1,\ldots,n_K$, minimize

$$L=-\log\sigma(s_{tc})-\sum_{i=1}^K\log\sigma(-s_{tn_i}).$$

Using $1-\sigma(s)=\sigma(-s)$, each negative term is the binary loss with label zero. Therefore

$$\nabla_{u_t}L=
\bigl(\sigma(s_{tc})-1\bigr)v_c
+\sum_{i=1}^K\sigma(s_{tn_i})v_{n_i}.$$

The positive contribution to its context row is

$$\bigl(\sigma(s_{tc})-1\bigr)u_t,$$

and each negative occurrence contributes

$$\sigma(s_{tn_i})u_t.$$

Repeated samples refer to the same parameter row. Add their contributions before updating that row. If a noise sample equals the observed context, its negative contribution also adds to that row's positive contribution; implementations may instead reject such collisions.

Only the center row and the observed/sampled context rows are needed, giving roughly $O((K+1)d)$ work per positive pair, excluding sampling overhead. This efficiency comes from changing the objective to observed-versus-noise classification. It does not produce a normalized $P(c\mid t)$ over the vocabulary.

## Why SGNS gives shifted PMI

### State the assumptions

Let $X_{tc}$ count observed word-context pairs, $X_t=\sum_cX_{tc}$, $X_c=\sum_tX_{tc}$, and $N=\sum_{t,c}X_{tc}$. These marginals all come from the **same pair-count table**. Draw $K$ negative contexts per positive pair from noise distribution $Q(c)$.

For this argument, treat every score $s_{tc}$ as an independently adjustable scalar and optimize the expected negative-sampling objective. Actual finite-dimensional word vectors couple these scores, so this is an idealized explanation rather than a guarantee for every trained embedding.

### Optimize a single pair's score

The expected positive count is $A=X_{tc}$. Across the $X_t$ positive pairs with center $t$, the expected negative count assigned to context $c$ is $B=KX_tQ(c)$.

That score's contribution to the log-likelihood is

$$F(s)=A\log\sigma(s)+B\log\sigma(-s).$$

For $A,B>0$:

$$F'(s)=A\bigl(1-\sigma(s)\bigr)-B\sigma(s).$$

At a stationary point,

$$\sigma(s^*)=\frac{A}{A+B},\qquad
e^{s^*}=\frac{\sigma(s^*)}{1-\sigma(s^*)}=\frac{A}{B}.$$

Thus

$$s^*_{tc}=\log\frac{X_{tc}}{KX_tQ(c)}.$$

This is the scalar maximum because

$$F''(s)=-(A+B)\sigma(s)\bigl(1-\sigma(s)\bigr)<0.$$

### Substitute unigram context noise

If the noise distribution equals the empirical context marginal, $Q(c)=X_c/N$, then

$$\begin{aligned}
s^*_{tc}
&=\log\frac{X_{tc}N}{X_tX_c}-\log K\\
&=\log\frac{P(t,c)}{P(t)P(c)}-\log K\\
&=\operatorname{PMI}_{\ln}(t,c)-\log K.
\end{aligned}$$

All logarithms here are natural logs. For a general noise distribution,

$$s^*_{tc}=\operatorname{PMI}_{\ln}(t,c)-\log K
+\log\frac{P(c)}{Q(c)}.$$

The usual $3/4$-power noise therefore changes the simple shifted-PMI expression. If $X_{tc}=0$ and $B>0$, the unconstrained optimum is approached as $s\to-\infty$ rather than at a finite value. Finite vector dimension, finite sampled data, and incomplete optimization further limit exact equality.

This explains a relationship with matrix factorization, but does not make SGNS identical to SVD of clipped PPMI. Their losses and constraints differ.

Return to [[natural_language_processing_cmpt_413/06_word2vec|the Word2vec lecture note]].
