# Logistic regression derivations

These derivations develop the probabilities, gradients, and regularization used in [[natural_language_processing_cmpt_413/03_logistic_regression|Logistic regression]]. Let $x\in\mathbb R^d$ be fixed features, $w\in\mathbb R^d$ weights, $b\in\mathbb R$ bias, and $y\in\{0,1\}$ the observed binary label. Logarithms are natural. Minimize losses; maximize log likelihoods.

## From log odds to sigmoid

For $p\in(0,1)$, odds of class 1 rather than class 0 are $p/(1-p)$. Probabilities .5, .8, and .2 correspond to odds 1, 4, and .25. Log odds map positive odds onto the real line:

$$
\ell=\log\frac{p}{1-p}.
$$

LR assumes these log-odds are linear in the features, $\ell=z=w^\top x+b$. Solve for the probability:

$$
\frac{p}{1-p}=e^z
\quad\Longrightarrow\quad
p=e^z(1-p)
\quad\Longrightarrow\quad
p(1+e^z)=e^z.
$$

Therefore:

$$
p=\frac{e^z}{1+e^z}=\frac1{1+e^{-z}}=\sigma(z).
$$

This gives a valid probability for every finite real logit. At $z=0$, $p=.5$; as $z\to\infty$, $p\to1$; as $z\to-\infty$, $p\to0$. Since sigmoid is strictly increasing, $p>\tau$ is equivalent to $z>\log[\tau/(1-\tau)]$. The .5 boundary is the hyperplane $w^\top x+b=0$.

For two-class softmax, divide numerator and denominator by $e^{z_1}$:

$$
p_1=\frac{e^{z_1}}{e^{z_0}+e^{z_1}}
=\frac1{1+e^{z_0-z_1}}
=\sigma(z_1-z_0).
$$

Only the logit difference matters. Subtracting a common constant $a$ from all softmax logits preserves probabilities, since $e^{-a}$ cancels. Taking $a=\max_cz_c$ helps numerical stability.

## Conditional likelihood and cross-entropy

For one binary example, write the observed-label probability as:

$$
P(y\mid x;w,b)=p^y(1-p)^{1-y}.
$$

When $y=1$ this selects $p$; when $y=0$ it selects $1-p$. Logarithms turn powers into coefficients:

$$
\log P(y\mid x;w,b)=y\log p+(1-y)\log(1-p).
$$

Maximizing it is equivalent to minimizing its negative:

$$
\ell(x,y;w,b)=-y\log p-(1-y)\log(1-p).
$$

This is binary cross-entropy. The observed label defines a target distribution assigning probability one to that label and zero to the other, so $-\sum_{k\in\{0,1\}}y_k\log p_k$ gives the same loss. This does not require knowing a complete population conditional label distribution.

For $N$ conditionally independent labeled examples given their inputs and model parameters:

$$
\mathcal L(w,b)=\prod_{i=1}^{N}p_i^{y_i}(1-p_i)^{1-y_i},
\qquad p_i=\sigma(w^\top x_i+b).
$$

The negative log becomes a sum:

$$
-\log\mathcal L(w,b)
=-\sum_{i=1}^{N}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

Divide by $N$ to obtain mean loss $J$. Without a penalty, this positive scaling preserves the optimum but scales gradients and effective learning rate. With a penalty, keep its strength relative to the data term consistent when switching between sum and mean.

For multiclass one-hot labels, $P(y_i\mid x_i)=\prod_cp_{ic}^{y_{ic}}$. Its negative log is $-\sum_cy_{ic}\log p_{ic}=-\log p_{i,\mathrm{true}}$.

## Sigmoid identities and the loss gradient

First, sigmoid symmetry gives:

$$
\sigma(-z)=\frac1{1+e^z}
=1-\frac{e^z}{1+e^z}
=1-\sigma(z).
$$

Next, differentiate $\sigma(z)=(1+e^{-z})^{-1}$:

$$
\sigma'(z)
=\frac{e^{-z}}{(1+e^{-z})^2}
=\frac1{1+e^{-z}}\frac{e^{-z}}{1+e^{-z}}
=\sigma(z)(1-\sigma(z)).
$$

Write $p=\sigma(z)$ and differentiate the loss with respect to probability:

$$
\frac{\partial\ell}{\partial p}=-\frac{y}{p}+\frac{1-y}{1-p}.
$$

The second term is positive because both the loss and the derivative of $1-p$ contribute a minus sign. Apply the chain rule:

$$
\begin{aligned}
\frac{\partial\ell}{\partial z}
&=\frac{\partial\ell}{\partial p}\frac{\partial p}{\partial z}\\
&=\left[-\frac{y}{p}+\frac{1-y}{1-p}\right]p(1-p)\\
&=-y(1-p)+(1-y)p\\
&=p-y.
\end{aligned}
$$

For finite logits, $p\in(0,1)$, so the divisions are defined. In implementations, computing the loss directly from logits avoids numerical issues near the limits.

For $z=\sum_jw_jx_j+b$, $\partial z/\partial w_j=x_j$ and $\partial z/\partial b=1$. Another chain-rule step gives:

$$
\frac{\partial\ell}{\partial w_j}=(p-y)x_j,\qquad
\nabla_w\ell=(p-y)x,\qquad
\frac{\partial\ell}{\partial b}=p-y.
$$

For the mean dataset loss:

$$
\nabla_wJ=\frac1N\sum_i(p_i-y_i)x_i,\qquad
\frac{\partial J}{\partial b}=\frac1N\sum_i(p_i-y_i).
$$

A mini-batch averages the same expressions over its examples; SGD uses one example. Calculate all derivatives at current parameters, then apply $w\leftarrow w-\eta\nabla_wJ$ and $b\leftarrow b-\eta\,\partial J/\partial b$ together.

For $y=1$, the residual $p-y$ is negative, so gradient descent pushes the observed example's logit upward. For $y=0$ it is positive and pushes it downward. At $w=(0,0)$, $b=0$, $x=(1,2)$, $y=1$, and $\eta=.1$, the weight gradient is $(-.5,-1)$, giving $w=(.05,.1)$ and $b=.05$. The logit rises to .30 and the loss decreases from $\log2\approx.693$ to $-\log\sigma(.30)\approx.554$.

## Why L1 produces sparse weights

For a differentiable data loss $L(w)$ and one scalar weight, compare:

$$
J_2(w)=L(w)+\lambda w^2,\qquad
J_1(w)=L(w)+\lambda|w|,\qquad \lambda\ge0.
$$

L2 has derivative $L'(w)+2\lambda w$. Its shrinkage term decreases continuously toward zero. At zero this term vanishes, so a nonzero $L'(0)$ generally moves the optimum away from zero. L2 can produce exact zeros in special cases but does not generally select them.

L1's derivative away from zero is:

$$
J_1'(w)=
\begin{cases}
L'(w)+\lambda,&w>0,\\
L'(w)-\lambda,&w<0.
\end{cases}
$$

The penalty supplies a constant pull toward zero on either side. It is nondifferentiable at zero; the subgradient there is $\partial|w|=[-1,1]$. For convex differentiable $L$, zero is optimal precisely when zero lies in the total objective's subgradient:

$$
0\in\partial J_1(0)=L'(0)+\lambda[-1,1]
\quad\Longleftrightarrow\quad
|L'(0)|\le\lambda.
$$

A whole range of weak data gradients can therefore be balanced at exactly zero. For a convex objective this guarantees a global minimum; for a general nonconvex objective it is only a stationarity condition. With multiple weights, the condition applies to each partial derivative evaluated at the jointly optimal parameters.

### A one-dimensional force example

Suppose the local data derivative is $L'(w)=-.3$, so the negative data gradient pushes rightward with force .3. Let $\lambda=1$.

If $w>0$, the total derivative is $-.3+1=.7$, so the negative gradient is $-.7$: it pulls left toward zero. If $w<0$, the derivative is $-.3-1=-1.3$, so the negative gradient is $+1.3$: it pushes right toward zero.

At zero, $|L'(0)|=.3\le1$. A penalty subgradient $+.3$ cancels the data derivative $-.3$, so zero satisfies optimality. The weak data signal does not justify a nonzero coefficient against this penalty.

This explains an optimum, not a guarantee that ordinary finite-step updates land exactly at zero: crossing zero can cause oscillation. L1-aware optimization must handle the nondifferentiable point appropriately.

### Geometry in more than one dimension

An equivalent view minimizes data loss subject to a weight-size constraint. In two dimensions, $|w_1|+|w_2|\le C$ is a diamond with corners on axes; $w_1^2+w_2^2\le C$ is a smooth disk. Expand a loss contour from the unconstrained minimum until it reaches the allowed region. L1 geometry often places the contact at a corner, where a coefficient is zero. L2 has no such corners, so contact typically gives smaller but nonzero coefficients.

Return to [[natural_language_processing_cmpt_413/03_logistic_regression|Logistic regression]] for the model, NLP prediction example, and training interpretation.
