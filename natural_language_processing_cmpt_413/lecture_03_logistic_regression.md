# Logistic regression

Logistic regression learns conditional class probabilities using linear scores, cross-entropy, gradient updates, and regularization. Neural layers introduce learned feature transformations.

## Binary probabilities

For features $x$, weights $w$, and bias $b$:

$$z=w^\top x+b,\qquad p=P(y=1\mid x)=\sigma(z)=\frac1{1+e^{-z}}.$$

The score $z$ is the log-odds $\log(p/(1-p))$. With threshold 0.5, the decision boundary is $w^\top x+b=0$: linear in the chosen features despite the nonlinear sigmoid.

Unlike Naive Bayes, LR models $P(y\mid x)$ directly rather than estimating how documents are generated given labels.

## Multiple classes

Give each class its own logit $z_c=w_c^\top x+b_c$, then normalize:

$$p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}.$$

Two-class softmax reduces to $\sigma(z_1-z_0)$. Subtracting the maximum logit before exponentiation preserves probabilities and avoids overflow.

## Loss and gradient

Binary cross-entropy is negative conditional log likelihood:

$$L=-y\log p-(1-y)\log(1-p).$$

For one-hot multiclass targets, the loss is $-\log p_{\mathrm{true}}$. The useful binary derivative is:

$$\frac{\partial L}{\partial z}=p-y,\qquad
\nabla_wL=(p-y)x,\qquad \frac{\partial L}{\partial b}=p-y.$$

Update parameters opposite the gradient: $\theta\leftarrow\theta-\eta\nabla L$. SGD uses sampled examples; mini-batches average a small group. The learning rate controls step size.

**Example:** with zero parameters, $x=(1,2)$ and $y=1$, $p=0.5$. A step of size 0.1 gives $w=(0.05,0.1)$ and $b=0.05$.

## Regularization and limits

Add $\lambda\lVert w\rVert_2^2$ to shrink weights or $\lambda\lVert w\rVert_1$ to encourage sparsity. Tune strength and thresholds on validation data.

The negative log likelihood is convex, but this does not guarantee a unique finite optimum: redundant features can cause nonuniqueness, and unregularized separable data can drive weights without bound.

## Introduction to learned neural features

A feedforward network alternates affine transformations with nonlinear activations:

$$h=f(W_1x+b_1),\qquad p=\operatorname{softmax}(W_2h+b_2).$$

For $d$ inputs, $H$ hidden units, and $K$ classes, $W_1$ is $H\times d$ and $W_2$ is $K\times H$. Check matrix shapes before counting parameters or differentiating.

Without nonlinear activations, composing affine layers gives another affine transformation. Hidden nonlinear transformations let the model learn features instead of relying only on the input features.
