# Logistic regression

Logistic regression learns conditional class probabilities from weighted input features. Its prediction function, cross-entropy objective, and gradient-based training also form building blocks for neural networks.

## A discriminative model with flexible features

For an input such as a document, extract a feature vector $x\in\mathbb R^d$. LR directly models $P(y\mid x)$ rather than estimating a class prior and a document-generation model as Naive Bayes does.

Features can include word counts, n-grams, sentiment lexicons, negation, parts of speech, parse information, and combinations of these. Feature templates generate combinations systematically; sparse indexing stores observed features without allocating every possible combination. Training data and representation determine which distinctions the classifier can learn.

LR does not require conditional independence of features. For overlapping features such as San, Francisco, and San Francisco, it can learn weights jointly instead of multiplying independent likelihoods. Duplicating a feature does not force extra evidence: its weights can sum to the original coefficient, although this can make parameters nonidentifiable and changes regularization. NB can remain competitive on small datasets; LR's flexibility does not guarantee better generalization.

## Binary probabilities, weights, and bias

For weights $w\in\mathbb R^d$ and scalar bias $b$:

$$
z=w^\top x+b,\qquad
p=P(y=1\mid x)=\sigma(z)=\frac1{1+e^{-z}},\qquad
P(y=0\mid x)=1-p.
$$

The linear score $z$ is the log-odds $\log[p/(1-p)]$. Increasing a positive-weight feature increases these log-odds, holding other features fixed; a negative weight decreases them. The bias supplies an intercept: at $x=0$, the model predicts $\sigma(b)$. Without a bias the .5 boundary must pass through the origin.

Predict class 1 when $p>.5$ and class 0 otherwise. The separating boundary is $w^\top x+b=0$, linear in the chosen features despite the nonlinear sigmoid. A tuned threshold $\tau\in(0,1)$ places it at $w^\top x+b=\log[\tau/(1-\tau)]$.

The [[natural_language_processing_cmpt_413/logistic_regression_derivations#From log odds to sigmoid|log-odds derivation]] explains the inverse transformation giving sigmoid.

### Sentiment features and a forward prediction

Suppose a review has these features:

| Feature | Meaning | Value |
|---|---|---:|
| $x_1$ | Positive lexicon word count | 3 |
| $x_2$ | Negative lexicon word count | 2 |
| $x_3$ | Contains no | 1 |
| $x_4$ | First- or second-person pronoun count | 3 |
| $x_5$ | Contains an exclamation mark | 0 |
| $x_6$ | Natural log of document word count | 4.15 |

Using $w=(2.5,-5,-1.2,.5,2,.7)$ and $b=.1$:

$$
z=2.5(3)-5(2)-1.2(1)+.5(3)+2(0)+.7(4.15)+.1=.805,
$$

$$
P(+\mid x)=\sigma(.805)\approx.691,\qquad P(-\mid x)\approx.309.
$$

Predict positive at a .5 threshold. The negative-word count contributes strongly against positive sentiment, but the other features offset it. Interpret coefficients alongside feature scale and correlated features; a weight is not an independent causal effect.

## Multiple classes and softmax

For $K$ classes, assign class-specific logits $z_c=w_c^\top x+b_c$ and normalize:

$$
p_c=P(y=c\mid x)=\frac{e^{z_c}}{\sum_{j=1}^{K}e^{z_j}}.
$$

These probabilities are positive and sum to one. Predict the largest probability, equivalently the largest logit. Each feature has a class-specific weight. For logits $(0,1,2,3,4)$, softmax gives approximately $(.0117,.0317,.0861,.2341,.6364)$.

Two-class softmax reduces to $p_1=\sigma(z_1-z_0)$: the logit difference is the binary log-odds. Adding a common constant to all logits preserves probabilities. Subtract the maximum logit before exponentiating to avoid overflow.

## Learn parameters with cross-entropy

For a binary label $y\in\{0,1\}$, the negative conditional log likelihood of one example is:

$$
\ell(x,y)=-y\log p-(1-y)\log(1-p).
$$

If $y=1$, loss is $-\log p$; if $y=0$, it is $-\log(1-p)$. The sentiment example's $p\approx.691$ gives loss about .370 for a positive label and 1.175 for a negative label. Loss approaches zero as the correct-label probability approaches one, and grows without bound for confidently wrong predictions.

Assuming training examples are conditionally independent under the model, maximize the product of their conditional label probabilities, equivalently minimize summed or averaged negative log likelihood:

$$
J(w,b)=\frac1N\sum_{i=1}^{N}\ell(x^{(i)},y^{(i)}).
$$

For multiclass one-hot targets, $\ell=-\sum_c y_c\log p_c=-\log p_{\mathrm{true}}$. This rewards probability assigned to the actual class, rather than merely a correct hard prediction.

The [[natural_language_processing_cmpt_413/logistic_regression_derivations#Conditional likelihood and cross-entropy|likelihood derivation]] shows how label probabilities produce this objective.

### Gradients and their interpretation

For one binary example:

$$
\frac{\partial\ell}{\partial z}=p-y,\qquad
\nabla_w\ell=(p-y)x,\qquad
\frac{\partial\ell}{\partial b}=p-y.
$$

For the mean objective, average these gradients over training examples. If $y=1$ but $p$ is low, $p-y<0$, and gradient descent increases the logit along the observed features. If $y=0$, $p-y>0$ and the update decreases it. A zero-valued feature contributes no gradient to its weight for that example; the bias still updates.

The [[natural_language_processing_cmpt_413/logistic_regression_derivations#Sigmoid identities and the loss gradient|chain-rule derivation]] includes the sigmoid derivative and intermediate simplification to $p-y$.

## Gradient descent and the training loop

LR has no general closed-form solution for maximum likelihood parameters. Optimize numerically, moving opposite the gradient:

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t),\qquad \theta=(w,b).
$$

The learning rate $\eta$ sets step size. Small values can make progress slow; large values can overshoot, oscillate, or increase loss. Select settings using validation data.

Full-batch gradient descent uses the whole training set for each update. Stochastic gradient descent uses one sampled example; mini-batch training averages a small group's gradients. These cheaper updates estimate the full gradient and are noisy. An epoch is one pass through the data.

Initialize parameters, compute logits and probabilities for a batch, evaluate loss, calculate gradients, update parameters, and repeat over batches and epochs. Compute all gradients at the same parameters used for the forward pass, then apply them together. Monitor validation performance for model selection and stopping; evaluate the final choice on held-out test data.

For example, start with $w=(0,0)$, $b=0$, $x=(1,2)$, and $y=1$. Then $p=.5$, $\nabla_w\ell=(-.5,-1)$, and $\partial\ell/\partial b=-.5$. With $\eta=.1$, one SGD step gives $w=(.05,.1)$ and $b=.05$. The new logit is .30, raising the probability to about .574.

Binary LR's negative log likelihood is convex in fixed-feature parameters: a local minimum is global. This does not ensure a unique finite optimum. Redundant features can create equivalent parameters; unregularized separable data can drive weights to infinity as loss approaches its infimum. Deep neural-network objectives are generally nonconvex.

## Regularization and generalization

Unrestricted fitting can learn noisy features and overfit. Add a penalty:

$$
J_{\mathrm{reg}}=J+\lambda R(w),\qquad \lambda\ge0.
$$

This is equivalent to subtracting the penalty when maximizing log likelihood. Specify whether bias is penalized; a common convention leaves it unpenalized. Also specify whether the data loss is a sum or mean, because that changes the relative penalty strength.

L2 uses $R(w)=\|w\|_2^2=\sum_jw_j^2$, adding gradient $2\lambda w$ and smoothly shrinking large weights. L1 uses $R(w)=\|w\|_1=\sum_j|w_j|$, encouraging exact zeros and sparse feature use. It is nondifferentiable at zero, requiring appropriate optimization.

An L1 constraint region has corners on coordinate axes, making zero coefficients common where a loss contour first touches it. A smooth L2 region tends to yield small coefficients instead. In one dimension, zero can be optimal for $L(w)+\lambda|w|$ when $|L'(0)|\le\lambda$: the penalty balances a weak data gradient exactly at zero. The [[natural_language_processing_cmpt_413/logistic_regression_derivations#Why L1 produces sparse weights|sparsity derivation and example]] develops this condition.

Tune penalty strength on validation data. Stronger regularization can improve generalization, but too much suppresses useful information.

## From logistic units to learned neural features

An artificial neuron computes a weighted sum plus bias, then an activation. Binary LR is a sigmoid neuron. Several parallel units produce a vector that can feed another layer; feedforward networks pass outputs forward without cycles.

A one-hidden-layer classifier computes:

$$
h=f(W_1x+b_1),\qquad p=\operatorname{softmax}(W_2h+b_2).
$$

For $d$ inputs, $H$ hidden units, and $K$ classes, $W_1$ has shape $H\times d$, $b_1$ length $H$, $W_2$ shape $K\times H$, and $b_2$ length $K$. Fully connected units receive every output of the preceding layer.

Sigmoid maps to $(0,1)$; tanh is zero-centered and maps to $(-1,1)$; ReLU computes $\max(0,z)$. Nonlinear hidden transformations learn representations and can create nonlinear boundaries in input space even when the final classifier is linear in hidden features. Without nonlinear activations, composing affine layers gives another affine transformation. Automatic differentiation libraries compute gradients through these building blocks.
