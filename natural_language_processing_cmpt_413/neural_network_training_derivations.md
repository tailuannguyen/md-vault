# Neural network training derivations

This note develops expressiveness and gradient calculations from [[natural_language_processing_cmpt_413/04_neural_networks|Neural networks]]. It keeps closely related foundational arguments together. Gradients below are for loss minimization; vectors are columns unless stated otherwise.

## Nonlinearity and constructive expressiveness

### Affine layers collapse without nonlinearities

Suppose two layers are $h=W_1x+b_1$ and $z=W_2h+b_2$. Substitution gives

$$z=W_2(W_1x+b_1)+b_2
=(W_2W_1)x+(W_2b_1+b_2).$$

Define $W'=W_2W_1$ and $b'=W_2b_1+b_2$. The result is one affine map $z=W'x+b'$. Repeating the argument collapses any sequence of affine layers. A sigmoid or softmax only at the output still produces a classifier based on these affine scores; nonlinear hidden transformations are what change the representational family.

### A hidden representation for XOR

On inputs $x_1,x_2\in\{0,1\}$, XOR returns 1 exactly when one input is 1. A single threshold classifier cannot separate the positive opposite corners from the negative opposite corners. Algebraically, negatives at $(0,0)$ and $(1,1)$ require $b\leq0$ and $w_1+w_2+b\leq0$, while positives require $w_1+b>0$ and $w_2+b>0$. Adding the positive inequalities gives $w_1+w_2+2b>0$, inconsistent with the negative inequalities.

Use the threshold activation $g(z)=1$ for $z>0$ and $0$ otherwise. Define

$$h_1=g(x_1+x_2),\qquad h_2=g(x_1+x_2-1),\qquad
\hat y=g(h_1-2h_2).$$

The first hidden unit computes OR and the second computes AND:

| Input $(x_1,x_2)$ | $h_1$ | $h_2$ | Output $\hat y$ |
|---|---|---|---|
| $(0,0)$ | 0 | 0 | 0 |
| $(0,1)$ | 1 | 0 | 1 |
| $(1,0)$ | 1 | 0 | 1 |
| $(1,1)$ | 1 | 1 | 0 |

The new coordinates make the classes separable by the output score $h_1-2h_2$. Convert to labels $\{-1,+1\}$ with $2\hat y-1$ if needed. This is a construction demonstrating capacity, not a gradient-training recipe for hard thresholds.

### Represent any Boolean function

Let $x\in\{-1,+1\}^D$ and choose a specific input pattern $a\in\{-1,+1\}^D$. Their dot product is $D$ when $x=a$; each differing coordinate lowers it by 2. Therefore

$$h_a(x)=g\bigl(a^\top x-(D-1)\bigr)$$

is 1 only for pattern $a$ and is 0 for every other Boolean input. With one hidden unit for each of the $2^D$ patterns, exactly one unit activates at any input.

For any target Boolean function $F$, choose the linear output

$$\hat F(x)=\sum_{a\in\{-1,+1\}^D}F(a)h_a(x).$$

Only $h_x$ contributes, giving $\hat F(x)=F(x)$. This explains the finite-input expressiveness claim while exposing its potentially exponential width. It is separate from continuous universal-approximation theorems and does not prove efficient learning or generalization.

## Local derivatives and graph traversal

### Chain rule at one operation

If $z=f(x)$ and the rest of the graph computes scalar $L$, then

$$\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial z}\frac{\partial z}{\partial x}.$$

The local derivative $\partial z/\partial x$ describes this operation; the incoming gradient $\partial L/\partial z$ describes its effect on the final loss. With multiple inputs, compute a gradient for each input. With multiple successor branches $z_j$, add their contributions:

$$\frac{\partial L}{\partial x}
=\sum_j\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial x}.$$

Run the forward pass in topological order and retain values needed for backward evaluation. Seed $\partial L/\partial L=1$ and process the graph in reverse order so each node has received its successors' contributions. Reuse computed gradients rather than separately expanding every full path.

### A multiply gate

For $z=xy$, the local derivatives are $\partial z/\partial x=y$ and $\partial z/\partial y=x$. Given incoming gradient $\bar z=\partial L/\partial z$:

$$\bar x=y\bar z,\qquad \bar y=x\bar z.$$

The backward operation needs the forward values of $x$ and $y$. For example, at $x=2,y=3$ with $\bar z=4$, it returns $\bar x=12$ and $\bar y=8$. A gate's backward calculation scales the local derivative; it does not assume the incoming gradient is always 1.

### A graph with a shared input

For $F(x,y,z)=(x+y)\max(y,z)$, introduce $a=x+y$, $b=\max(y,z)$, and $F=ab$. At $x=1,y=2,z=0$:

$$a=3,\qquad b=2,\qquad F=6.$$

Starting with $\bar F=1$, multiplication sends $\bar a=b=2$ and $\bar b=a=3$. Addition sends 2 to both $x$ and $y$. Since $y>z$, the maximum sends its entire gradient 3 to $y$ and zero to $z$. Thus

$$\frac{\partial F}{\partial x}=2,\qquad
\frac{\partial F}{\partial y}=2+3=5,\qquad
\frac{\partial F}{\partial z}=0.$$

The two contributions to $y$ must accumulate rather than overwrite one another. At a tie $y=z$, maximum is not differentiable; a chosen backward convention is needed. This example avoids that case.

## Loss gradients and matrix backpropagation

### Sigmoid with binary cross-entropy

Let $z=w^\top x+b$, $p=\sigma(z)$, and target $y\in\{0,1\}$. For

$$L=-y\log p-(1-y)\log(1-p),$$

first differentiate the loss with respect to its probability input:

$$\begin{aligned}
\frac{\partial L}{\partial p}
&=-\frac{y}{p}+\frac{1-y}{1-p}\\
&=\frac{-y(1-p)+(1-y)p}{p(1-p)}\\
&=\frac{p-y}{p(1-p)}.
\end{aligned}$$

For finite $z$, $0<p<1$ and $\partial p/\partial z=p(1-p)$. The chain rule cancels the denominator:

$$\frac{\partial L}{\partial z}
=\frac{p-y}{p(1-p)}p(1-p)=p-y.$$

The affine operation has $\nabla_wz=x$, $\partial z/\partial b=1$, and $\nabla_xz=w$, hence

$$\nabla_wL=(p-y)x,\quad
\frac{\partial L}{\partial b}=p-y,\quad
\nabla_xL=(p-y)w.$$

The score gradient is prediction minus target. A positive example with too-small probability has $p-y<0$, so gradient descent raises its score. Stable implementations compute this loss directly from logits rather than evaluating logs of numerically rounded probabilities.

### Softmax with categorical cross-entropy

For logits $z_k$, $p_k=e^{z_k}/Z$, and $Z=\sum_j e^{z_j}$, a one-hot target $t$ satisfies $\sum_k t_k=1$. Rewrite

$$L=-\sum_k t_k\log p_k
=-\sum_k t_kz_k+\log Z.$$

For each coordinate $j$:

$$\frac{\partial L}{\partial z_j}
=-t_j+\frac1Z\frac{\partial Z}{\partial z_j}
=-t_j+\frac{e^{z_j}}Z=p_j-t_j.$$

The same result holds for a target probability distribution summing to 1. Its gradient coordinates sum to zero because both $p$ and $t$ sum to 1.

### Apply the chain rule to a two-layer classifier

This calculation applies the graph rules to the matrix network in the main note, making the parameter gradients explicit. Let

$$z_1=W_1x+b_1,\quad h=f(z_1),\quad
z_2=W_2h+b_2,\quad p=\operatorname{softmax}(z_2).$$

With $d$ inputs, $H$ hidden units, and $K$ classes, the output error is

$$\delta_2=\nabla_{z_2}L=p-t\in\mathbb R^K.$$

Each output score is $z_{2,k}=\sum_jW_{2,kj}h_j+b_{2,k}$. Therefore $\partial L/\partial W_{2,kj}=\delta_{2,k}h_j$, giving the outer product

$$\nabla_{W_2}L=\delta_2h^\top\in\mathbb R^{K\times H},\qquad
\nabla_{b_2}L=\delta_2.$$

Each hidden unit affects all output scores, so branch accumulation gives $\nabla_hL=W_2^\top\delta_2$. The elementwise activation then gives

$$\delta_1=(W_2^\top\delta_2)\odot f'(z_1)\in\mathbb R^H.$$

Apply the affine rule again:

$$\nabla_{W_1}L=\delta_1x^\top\in\mathbb R^{H\times d},\qquad
\nabla_{b_1}L=\delta_1,\qquad
\nabla_xL=W_1^\top\delta_1.$$

These equations show how an output error reaches earlier parameters. Compute all gradients using the parameters from the same forward pass, then let the optimizer update them. For a minibatch-mean loss, average the per-example gradients; parameter sharing also requires summing every contribution. If the objective adds a regularizer, add its gradient separately.

Return to [[natural_language_processing_cmpt_413/04_neural_networks|Neural networks]] for optimization and generalization.
