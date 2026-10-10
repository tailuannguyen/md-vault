# Neural networks

Neural networks learn features alongside a predictor. Forward computation produces predictions, a loss measures error, backpropagation computes gradients, and an optimizer changes the parameters. These are separate parts of training.

## Layers and expressiveness

A neuron applies a nonlinear activation to an affine score, $f(w^\top x+b)$. A feedforward classifier combines such units into layers:

$$z_1=W_1x+b_1,\qquad h=f(z_1),\qquad
z_2=W_2h+b_2,\qquad p=\operatorname{softmax}(z_2).$$

For input dimension $d$, hidden width $H$, and $K$ classes, use column vectors and shapes

$$x\in\mathbb R^d,\quad W_1\in\mathbb R^{H\times d},\quad
b_1,h\in\mathbb R^H,\quad W_2\in\mathbb R^{K\times H},\quad
b_2,p\in\mathbb R^K.$$

The hidden activation acts elementwise. This network has $Hd+H+KH+K$ trainable parameters. More hidden layers repeat $h^{(l)}=f(W_lh^{(l-1)}+b_l)$ with compatible dimensions; the output activation depends on the task.

Without nonlinear activations, composing affine layers gives another affine transformation. Depth alone then cannot create nonlinear decision boundaries. XOR illustrates the limitation: $(0,0)$ and $(1,1)$ have one label, while $(0,1)$ and $(1,0)$ have the other, so no single line separates the classes. Two nonlinear hidden units can transform these inputs into a space where a linear output separates them.

With sufficient capacity, nonlinear networks can approximate broad classes of functions. For Boolean inputs, one threshold hidden unit per possible input configuration gives a concrete construction with $2^D$ units for $D$ inputs. Deeper networks can represent some functions more compactly. Expressiveness describes functions a model can represent; it does not guarantee that optimization finds the desired parameters or that the model generalizes.

The [[natural_language_processing_cmpt_413/neural_network_training_derivations#Nonlinearity and constructive expressiveness|affine-composition argument, worked XOR network, and Boolean construction]] develop these claims explicitly.

## Activations

| Activation | Definition | Derivative away from nondifferentiable points | Main consideration |
|---|---|---|---|
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z))$ | Bounded in $(0,1)$; saturates at large magnitudes |
| Tanh | $\tanh z$ | $1-\tanh^2z$ | Zero-centered, but also saturates |
| ReLU | $\max(0,z)$ | $1$ for $z>0$, $0$ for $z<0$ | Simple; a unit can become inactive with zero gradient |
| Leaky ReLU | $z$ for $z\geq0$, $az$ otherwise | $1$ or $a$ | Small positive $a$ preserves a negative-side gradient |

ReLU's derivative at zero requires an implementation convention. Saturated sigmoid/tanh units have small local derivatives, which can weaken gradients passed through many layers. ReLU avoids positive-side saturation but can produce dead units; leaky ReLU addresses its zero negative-side derivative.

There is no universally best activation. Other families include ELU, PReLU, and Swish; some introduce trainable shape parameters. Hidden activations control the learned representation, whereas output activations impose the form of predictions.

## Match outputs and losses to the task

For a binary target $y\in\{0,1\}$, predict $p=\sigma(z)$ and minimize binary cross-entropy:

$$L=-y\log p-(1-y)\log(1-p).$$

For $K$ mutually exclusive classes, logits $z\in\mathbb R^K$ become a distribution:

$$p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\qquad
L=-\sum_k t_k\log p_k=-\log p_y,$$

where $t$ is the one-hot target vector and $y$ is the correct class. The probabilities are nonnegative and sum to one.

For scalar regression, use an unrestricted linear output $\hat y=w^\top h+b$ and, for example, squared error $L=(\hat y-y)^2$. Dataset objectives usually sum or average these per-example losses. A network architecture does not determine its loss automatically: choose the prediction space and objective together.

## Computational graphs and backpropagation

Represent computation as a directed acyclic graph: nodes are operations, and edges carry values. Visit nodes in topological order for the forward pass, storing values needed by local derivatives. Start the scalar loss with $\partial L/\partial L=1$, then visit nodes in reverse order.

At each node, multiply the incoming loss gradient by the operation's local derivative. If a value $x$ affects multiple successors $z_j$, accumulate all paths:

$$\frac{\partial L}{\partial x}
=\sum_j\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial x}.$$

For $f=(x+y)\max(y,z)$ at $(1,2,0)$, the forward value is 6 and the gradient is $(2,5,0)$. The $y$ gradient is $2+3=5$ because $y$ affects both addition and maximum. The [[natural_language_processing_cmpt_413/neural_network_training_derivations#Local derivatives and graph traversal|worked graph and multiply gate]] show the complete backward calculation.

For sigmoid with binary cross-entropy, the chain rule simplifies to

$$\frac{\partial L}{\partial z}=p-y,\qquad
\nabla_wL=(p-y)x,\qquad \frac{\partial L}{\partial b}=p-y.$$

For softmax with categorical cross-entropy, the logit gradient is $p-t$. A hidden layer passes this error through its weights and multiplies it elementwise by its activation derivative. The [[natural_language_processing_cmpt_413/neural_network_training_derivations#Loss gradients and matrix backpropagation|loss-gradient derivations and two-layer matrix calculation]] explain the intermediate steps and dimensions.

Backpropagation computes derivatives; it does not update weights. Automatic differentiation applies this process to the graph actually implemented, so choosing correct operations, outputs, and losses remains necessary.

## Optimization

Let $J(\theta)=\frac1N\sum_i L_i(\theta)$ and let $g_t$ be the gradient used at update $t$. Gradient descent uses the entire dataset. SGD samples one example; minibatch SGD uses an average over a batch $B_t$:

$$g_t=\frac1{|B_t|}\sum_{i\in B_t}\nabla_\theta L_i(\theta_t),\qquad
\theta_{t+1}=\theta_t-\eta_tg_t.$$

Uniform sampling gives an unbiased estimate of the full average gradient. Minibatches generally reduce sampling variance and make matrix computation more efficient than one-example updates. Shuffle examples during training; choose a learning rate and optionally decay it over time. Each training step performs forward computation, loss evaluation, gradient calculation, and an optimizer update. Clear old gradient accumulations when they are not intentionally being retained.

**Momentum** adds a velocity that retains previous update directions:

$$v_t=\mu v_{t-1}-\eta_tg_t,\qquad
\theta_{t+1}=\theta_t+v_t,\quad 0\leq\mu<1.$$

Initialize $v_0=0$. Persistent directions build up, while inconsistent directions can cancel, reducing oscillation.

**AdaGrad** accumulates squared gradients coordinatewise:

$$s_t=s_{t-1}+g_t\odot g_t,\qquad
\theta_{t+1}=\theta_t-\eta_t\frac{g_t}{\sqrt{s_t}+\epsilon}.$$

Large accumulated gradients reduce a coordinate's effective step size. Initialize $s_0=0$; square roots and divisions are elementwise, and $\epsilon>0$ prevents division by zero. Accumulation also means effective learning rates can become very small late in training.

**Adam** uses exponential moving averages of gradients and their squares:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
s_t=\beta_2s_{t-1}+(1-\beta_2)(g_t\odot g_t),$$

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat s_t=\frac{s_t}{1-\beta_2^t},\qquad
\theta_{t+1}=\theta_t-\eta_t\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}.$$

Initialize both averages to zero; the hats correct their initial bias. Adam combines a smoothed update direction with per-coordinate scaling. Its second moment is the average squared gradient, not directly the centered statistical variance. **AdamW** decouples weight decay from the adaptive gradient update, applying a separate shrinkage term such as $-\eta_t\lambda\theta_t$ to parameters selected for decay.

Neural-network objectives are generally nonconvex. These methods help optimization, but do not guarantee finding a global minimum.

## Generalization and regularization

A model can fit training data, including noise, while performing poorly on unseen data. Monitor validation performance separately from the training objective.

- **Weight penalties:** optimize $J_{\mathrm{reg}}=J+\lambda R(\theta)$, for example an L1 or L2 penalty. The penalty discourages particular parameter configurations; it does not simply add more data.
- **Dropout:** randomly mask activations during training so prediction cannot depend exclusively on a small set of units. Random masking is disabled at evaluation, using the framework's corresponding scaling convention.
- **Early stopping:** retain parameters from an appropriate validation checkpoint when additional training worsens validation performance. Use the test set for final evaluation rather than choosing the stopping point.

Deep-learning frameworks supply layers, losses, optimizers, automatic differentiation, batching, and GPU computation. They reduce implementation work but do not replace decisions about architecture, data, and evaluation.

## Other supervised classifiers

Neural networks learn hidden features; logistic regression typically combines supplied features with a linear score. Naive Bayes estimates a generative probability model from counts, whereas logistic regression models the conditional class distribution.

An **SVM** chooses a separating hyperplane with a large margin. For labels $y_i\in\{-1,+1\}$ and score $s_i=w^\top f(x_i)-b$, a soft-margin objective is

$$\frac1N\sum_i\max(0,1-y_is_i)+\lambda\lVert w\rVert^2.$$

The hinge loss is zero once the correct signed score reaches the margin of 1; otherwise it penalizes margin violations. Kernels allow a linear rule in an implicit feature space. A multiclass variant sums $\max(0,s_k-s_y+1)$ over incorrect classes $k$.

**kNN** labels a new example by its nearest training examples under a selected distance or similarity, typically using a majority vote. It needs no fitted weight optimization but must retain training data. Choice of $k$ and metric matters; inference can require considerable memory and search time, motivating approximate nearest-neighbor methods.
