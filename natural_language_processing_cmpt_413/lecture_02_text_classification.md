# Text classification and Naive Bayes

Text classification maps document features to labels. Naive Bayes combines class priors with word likelihoods, and evaluation measures the errors relevant to the task.

## Model a document

Choose labels, a feature representation, a model, and a training objective. Train on labeled documents, then apply the learned classifier to new inputs.

Bag of words represents a document using word counts and discards order. Binary features record presence rather than repetitions. N-gram features can retain short phrases such as “not good.” Tokenization and feature extraction must be consistent across training and prediction.

## Naive Bayes

Bayes' rule gives $P(c\mid d)\propto P(c)P(d\mid c)$. Naive Bayes assumes word observations are conditionally independent given the class. Its stable log score is:

$$\operatorname{score}(c,d)=\log P(c)+\sum_{w\in V}C_d(w)\log P(w\mid c).$$

Choose the class with the largest score. The evidence $P(d)$ is constant across classes, so it cancels for ranking.

With $N_c$ class documents, $N$ total documents, $T_c$ class tokens, and word count $C(w,c)$:

$$\hat P(c)=N_c/N,\qquad
\hat P(w\mid c)=\frac{C(w,c)+\alpha}{T_c+\alpha\lvert V\rvert}.$$

The class prior counts **documents**; the word likelihood counts **tokens**. Smoothing prevents one unseen word-class pair from zeroing the whole document score.

## Choose the event model

- **Multinomial NB:** repeated word occurrences contribute repeatedly.
- **Binary multinomial NB:** each present word contributes once.
- **Bernoulli NB:** both presence and absence contribute; estimate presence from documents containing the word.

Binarizing a multinomial input does not make it a Bernoulli likelihood. For example, 80 occurrences in one email still mean presence in only one document.

## What to remember

NB is a useful small-data baseline. Correlated evidence can be counted repeatedly, and its assumptions need not match language. Logistic regression directly models class probabilities and permits overlapping features without this independence assumption.

Use classification metrics and held-out splits to judge performance rather than assuming the more complex model always wins.

## Data splits and model selection

Train parameters and preprocessing statistics on training data. Use validation data for smoothing, learning rates, penalties, thresholds, or checkpoint selection. Reserve test data for evaluation after those decisions.

Repeated model selection using test scores leaks information into the chosen system even without gradient training on the test examples.

## Classification metrics

Label confusion-matrix axes before extracting TP, FP, FN, and TN:

$$\mathrm{Accuracy}=\frac{TP+TN}{N},\quad
P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad
F_1=\frac{2TP}{2TP+FP+FN}.$$

Precision measures correctness among selected positives; recall measures recovery of actual positives. Accuracy can hide minority-class errors.

Macro F1 averages class-level F1 values. Micro F1 pools counts first, giving common classes more influence. Choose thresholds on validation data according to task needs.

**Example:** with $TP=50,FP=25,FN=25,TN=150$, accuracy is 0.8 but precision, recall, and F1 are each $2/3$.
