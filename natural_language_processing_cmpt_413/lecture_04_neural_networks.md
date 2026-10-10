# Neural networks

Neural networks learn nonlinear features. Forward computation produces predictions, backpropagation computes gradients, and an optimizer updates parameters.

## Layers and expressiveness

A feedforward network alternates affine transformations with nonlinear activations:

$$h=f(W_1x+b_1),\qquad p=\operatorname{softmax}(W_2h+b_2).$$

For $d$ inputs, $H$ hidden units, and $K$ classes, $W_1$ is $H\times d$ and $W_2$ is $K\times H$. Check matrix shapes before counting parameters or differentiating.

Without nonlinear activations, composing affine layers gives another affine transformation. XOR illustrates the need for nonlinearity: its opposite-corner labels cannot be separated by one line, but a hidden representation can make them separable. Universal approximation describes representational capacity, not a guarantee of successful training.

## Activations

- **Sigmoid:** bounded in $(0,1)$; derivative $p(1-p)$.
- **Tanh:** zero-centered and bounded; derivative $1-\tanh^2z$.
- **ReLU:** $\max(0,z)$; simple, but negative inputs receive zero derivative.
- **Leaky ReLU:** keeps a small negative-side slope.

Sigmoid and tanh can saturate, weakening gradients. Choose hidden activations separately from output activations: output probabilities and losses must match the task.

## Backpropagation

Compute values forward through the graph; traverse operations backward using the chain rule. If a value affects several branches, add all derivative contributions.

For $f=(x+y)\max(y,z)$ at $(1,2,0)$, the forward value is 6 and the gradient is $(2,5,0)$. The $y$ derivative is 5 because it receives contributions through both addition and maximum.

Backpropagation computes derivatives; it does not update weights. Automatic differentiation performs these operations on the graph you implemented.

## Optimization and generalization

SGD updates opposite a sampled gradient. Momentum retains update history. AdaGrad scales coordinates using accumulated squared gradients. Adam uses decayed first and second moments; AdamW decouples weight decay from its adaptive update.

L1/L2 penalties, dropout, and validation-based early stopping help limit overfitting. Dropout randomly masks activations during training and is disabled during evaluation. Neural objectives are generally nonconvex.

Other classifier approaches include SVM's margin objective and kNN's neighbor voting.
