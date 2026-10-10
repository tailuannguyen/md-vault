# Language model derivations

These derivations develop the count estimates and sentence normalization used in [[natural_language_processing_cmpt_413/01b_language_models|Language models]]. Optimization uses natural logarithms; another base multiplies the objective by a positive constant and leaves its maximizer unchanged.

## Maximum likelihood for a categorical distribution

Let $\mathcal W$ be a finite set of token outcomes. Parameters $p_w=P(w)$ satisfy $p_w\geq0$ and $\sum_wp_w=1$. Suppose $N>0$ observed draws are independent under this model. Let $C(w)$ count outcome $w$, so $\sum_wC(w)=N$.

Group equal factors in the likelihood:

$$
L(p)=\prod_{i=1}^{N}p_{w_i}=\prod_wp_w^{C(w)},\qquad
\log L(p)=\sum_wC(w)\log p_w.
$$

The likelihood treats the observed data as fixed and varies the parameters. It is not itself a probability distribution over parameters.

First restrict to outcomes with positive counts. Their probabilities must be positive at an optimum, since zero would give zero likelihood. Unobserved outcomes receive zero mass at the MLE: mass allocated there could instead increase observed probabilities and raise likelihood. Smoothing changes this boundary behavior.

Introduce a multiplier $\lambda$ for normalization:

$$
\mathcal L(p,\lambda)=\sum_wC(w)\log p_w
-\lambda\left(\sum_wp_w-1\right).
$$

For positive-count outcomes, differentiate and set the derivative to zero:

$$
\frac{\partial\mathcal L}{\partial p_w}
=\frac{C(w)}{p_w}-\lambda=0
\quad\Longrightarrow\quad p_w=\frac{C(w)}\lambda.
$$

Use normalization to determine $\lambda$:

$$
1=\sum_wp_w=\frac1\lambda\sum_wC(w)=\frac N\lambda
\quad\Longrightarrow\quad\lambda=N.
$$

Therefore:

$$
\hat p_w=\frac{C(w)}N.
$$

This is the sample mean of the indicator $\mathbf1[w_i=w]$. Counts $(2,1,1)$ give probabilities $(1/2,1/4,1/4)$.

To confirm a maximum rather than only a stationary point, on positive-count coordinates:

$$
\frac{\partial^2\log L}{\partial p_w^2}=-\frac{C(w)}{p_w^2}<0,
$$

with zero cross-derivatives. The objective is concave on the probability simplex, so the stationary solution on the observed support is its global maximum. Zero-count coordinates are handled by the boundary argument above. If $N=0$, the data provide no unique count estimate.

### Apply the result to unigrams

A unigram model predicts every token from the same distribution, independent of preceding tokens. For sentences $s^{(j)}=(w_1^{(j)},\ldots,w_{T_j}^{(j)})$:

$$
\log L(p)=\sum_j\sum_{i=1}^{T_j}\log p_{w_i^{(j)}}
=\sum_wC(w)\log p_w.
$$

Hence $\hat P(w)=C(w)/N$, with $N=\sum_jT_j$. If predicting STOP, append it once per sentence and include those events in both $C$ and $N$. Fixed start markers provide context and are not additional predicted outcomes. Counting must agree with the sequence model being fitted.

## Conditional maximum likelihood for bigrams

Let $p_{u,w}=P(w\mid u)$ be the next-token distribution for preceding-token context $u$. Each row has its **own** normalization constraint:

$$
p_{u,w}\geq0,\qquad\sum_wp_{u,w}=1\quad\text{for every }u.
$$

Pad training sentences with their initial context and include STOP predictions when modeling complete sentences. Assuming independent sentences and the bigram Markov approximation:

$$
\log L(p)=\sum_j\sum_i\log p_{w_{i-1}^{(j)},w_i^{(j)}}
=\sum_u\sum_wC(u,w)\log p_{u,w}.
$$

$C(u,w)$ counts modeled transitions, and $C(u)=\sum_wC(u,w)$ counts predictions from context $u$. This avoids counting word occurrences with no successor under the selected boundary convention.

The objective separates by history. Use one multiplier $\lambda_u$ per row:

$$
\mathcal L(p,\lambda)=\sum_u\sum_wC(u,w)\log p_{u,w}
-\sum_u\lambda_u\left(\sum_wp_{u,w}-1\right).
$$

For $C(u,w)>0$:

$$
\frac{\partial\mathcal L}{\partial p_{u,w}}
=\frac{C(u,w)}{p_{u,w}}-\lambda_u=0
\quad\Longrightarrow\quad p_{u,w}=\frac{C(u,w)}{\lambda_u}.
$$

Normalize that row:

$$
1=\sum_wp_{u,w}=\frac{\sum_wC(u,w)}{\lambda_u}
\quad\Longrightarrow\quad\lambda_u=C(u).
$$

Thus:

$$
\hat P(w\mid u)=\frac{C(u,w)}{C(u)},\qquad C(u)>0.
$$

Each row maximizes a concave categorical log-likelihood, so together these estimates maximize the bigram likelihood. Unseen successors in an observed row receive zero probability. For an entirely unseen context, $C(u)=0$ and the data do not constrain its distribution: $0/0$ is not an estimate. Smoothing or a fallback defines that row.

The proof extends to longer n-grams by replacing $u$ with history $h$, with separate normalization for each history and $\hat P(w\mid h)=C(h,w)/C(h)$.

## Normalization with a STOP symbol

Let $V$ contain ordinary tokens and $V^*$ every finite sequence over $V$, including the empty sequence. A model of complete sentences should satisfy $\sum_{s\in V^*}P(s)=1$.

### Why local normalization without stopping is insufficient

With unigram probabilities $P(a)=P(b)=1/2$, already $P(a)+P(b)+P(aa)+P(bb)=3/2$. More generally, at every fixed length $n$:

$$
\sum_{w_1,\ldots,w_n\in V}\prod_{i=1}^{n}P(w_i)
=\prod_{i=1}^{n}\sum_{w_i\in V}P(w_i)=1.
$$

Summing this mass over infinitely many lengths diverges. These products are valid fixed-length or prefix probabilities, but cannot simultaneously represent complete sentences of arbitrary length without modeling termination or length.

### Normalize complete unigram sentences

Extend the outcome set to $V\cup\{\mathrm{STOP}\}$. Let $q=P(\mathrm{STOP})$, with $0<q\leq1$, so $\sum_{w\in V}P(w)=1-q$. A sentence with $n$ ordinary tokens has probability:

$$
P(w_1\cdots w_n\mathrm{STOP})=\left[\prod_{i=1}^{n}P(w_i)\right]q.
$$

Sum over all length-$n$ sentences:

$$
\begin{aligned}
P(\text{length}=n)
&=q\sum_{w_1\in V}\cdots\sum_{w_n\in V}\prod_{i=1}^{n}P(w_i)\\
&=q\prod_{i=1}^{n}\left[\sum_{w_i\in V}P(w_i)\right]\\
&=q(1-q)^n.
\end{aligned}
$$

The sums factor because unigram probabilities do not depend on preceding tokens. Apply the geometric-series identity:

$$
\begin{aligned}
\sum_{s\in V^*}P(s)
&=\sum_{n=0}^{\infty}q(1-q)^n\\
&=q\frac1{1-(1-q)}=1.
\end{aligned}
$$

For $q=1$, only the empty sentence has positive mass; its empty product is one. For $q=0$, termination never occurs, so finite completed sentences receive no mass.

For example, $P(a)=P(b)=1/4$ and $q=1/2$ give $P(\text{empty})=1/2$, $P(a\mathrm{STOP})=1/8$, and $P(aa\mathrm{STOP})=1/32$. Excluding the empty sentence leaves mass $1-q$; conditioning on nonempty output normalizes this smaller event space.

### Context-dependent models

For an n-gram or neural model, STOP probability can depend on history, so the geometric unigram formula need not apply. Total mass on finite complete sequences equals the probability that generation eventually stops. A proper finite-sentence model requires that probability to be one.

One sufficient condition is $P(\mathrm{STOP}\mid h)\geq\varepsilon>0$ for every reachable nonterminated history. The probability of surviving $n$ predictions is at most $(1-\varepsilon)^n$, which tends to zero. Merely including STOP is insufficient if a reachable state can continue forever without stopping.

Return to [[natural_language_processing_cmpt_413/01b_language_models|Language models]] for smoothing, generation, and evaluation.
