# Text classification and Naive Bayes

Text classification assigns a document or sentence to a class. Spam detection, sentiment analysis, authorship attribution, language identification, and news categorization are examples. Classification also appears inside larger NLP systems: predicting the next word selects a vocabulary item, intent detection chooses an action, and parsing chooses word attachments.

## Define the task and representation

For classes $C=\{c_1,\ldots,c_K\}$ and labeled training documents $\{(d_i,y_i)\}_{i=1}^N$, learn a classifier $F:d\mapsto c$. Binary classification has two classes; multiclass classification has more. Supervised learning uses labeled examples; unsupervised learning seeks structure without these labels. Classification predicts a category, while regression predicts a numerical value.

A rule-based classifier matches explicit patterns, such as positive sentiment words or suspicious email domains. Carefully chosen rules can be accurate, but defining and maintaining them is expensive, and rules may generalize poorly. Statistical classifiers learn which observed patterns support each label.

Building a learned model involves an input representation, a prediction function, a training objective, and a learning algorithm. Training estimates parameters from labeled inputs; prediction applies those parameters to a new input.

Bag of words represents a document by counts $C_d(w)$ over a vocabulary $V$. It discards word positions, so “dog bites man” and “man bites dog” have the same counts. Binary features record presence rather than repetitions. N-grams retain short phrases such as “not good”; negation features such as NOT_happy can distinguish negative context from happy alone. Features can also include URLs, email domains, capitalization, and linguistic information. Useful representations depend on the task and domain.

## Naive Bayes: from Bayes' rule to a classifier

Bayes' rule relates a posterior to a prior and a document likelihood:

$$
P(c\mid d)=\frac{P(c)P(d\mid c)}{P(d)}.
$$

The maximum a posteriori prediction is $\hat c=\arg\max_{c\in C}P(c\mid d)$. The evidence $P(d)$ is constant across candidate classes, so it can be omitted when ranking them.

For a document $d=(w_1,\ldots,w_M)$, Naive Bayes assumes word observations are conditionally independent given the class and their probabilities do not depend on position:

$$
P(d\mid c)=\prod_{i=1}^{M}P(w_i\mid c).
$$

This is a modeling assumption, not a claim about independence in actual language. Taking logarithms preserves the ranking and turns products into sums:

$$
\hat c
=\arg\max_c\left[\log P(c)+\sum_{i=1}^{M}\log P(w_i\mid c)\right]
=\arg\max_c\left[\log P(c)+\sum_{w\in V}C_d(w)\log P(w\mid c)\right].
$$

The second expression groups repeated occurrences. Log scores avoid underflow from multiplying many small probabilities. They rank classes but are not themselves normalized posterior probabilities.

### Estimate parameters and smooth sparse counts

Let $N_c$ be the number of class-$c$ training documents, $C(w,c)$ the total occurrences of $w$ in those documents, and $T_c=\sum_{w\in V}C(w,c)$ their total token count. Maximum likelihood estimates are:

$$
\hat P(c)=\frac{N_c}{N},\qquad
\hat P(w\mid c)=\frac{C(w,c)}{T_c}.
$$

The prior counts **documents**; the word likelihood counts **tokens**. For example, “good food good service” and “good price” contain three occurrences of good in six tokens, giving an unsmoothed likelihood of $3/6$.

Rare words create sparse counts. An unseen word-class pair has zero likelihood without smoothing, making the entire document likelihood zero for that class. Add a pseudocount $\alpha>0$ to every vocabulary word:

$$
\hat P(w\mid c)=\frac{C(w,c)+\alpha}{T_c+\alpha|V|}.
$$

The denominator adds $\alpha$ for each vocabulary word, preserving normalization. $\alpha=1$ is add-one or Laplace smoothing. Larger values pull estimates more strongly toward uniform probabilities; select the value using validation data.

Training builds the vocabulary from training data, counts class documents and class-specific tokens, and estimates probabilities. Prediction tokenizes consistently and chooses the largest log score.

### Worked training and prediction example

Consider:

| Document | Tokens | Class |
|---|---|---|
| 1 | Chinese Beijing Chinese | $c$ |
| 2 | Chinese Chinese Shanghai | $c$ |
| 3 | Chinese Macao | $c$ |
| 4 | Tokyo Japan Chinese | $j$ |

The vocabulary has six words: Chinese, Beijing, Shanghai, Macao, Tokyo, and Japan. Priors are $P(c)=3/4$ and $P(j)=1/4$. Class $c$ has eight tokens, including five occurrences of Chinese; class $j$ has three tokens.

With $\alpha=1$:

$$
P(\text{Chinese}\mid c)=\frac{5+1}{8+6}=\frac37,\qquad
P(\text{Tokyo}\mid c)=P(\text{Japan}\mid c)=\frac1{14}.
$$

Chinese, Tokyo, and Japan each occur once in class $j$, giving likelihood $(1+1)/(3+6)=2/9$ each. For “Chinese Chinese Chinese Tokyo Japan,” compare unnormalized scores:

$$
S_c=\frac34\left(\frac37\right)^3\left(\frac1{14}\right)^2\approx0.0003012,\qquad
S_j=\frac14\left(\frac29\right)^5\approx0.0001355.
$$

Predict $c$ because $S_c>S_j$. Repeated Chinese tokens contribute three times; smoothing permits the previously unseen Tokyo and Japan tokens in class $c$. Normalize if posterior probabilities are needed: $P(c\mid d)=S_c/(S_c+S_j)\approx.690$.

## Choose the event model

**Multinomial NB** uses ordinary token counts. Repeated words contribute repeatedly during training and prediction.

**Binary multinomial NB** clips each document's counts to presence indicators $I_d(w)=\mathbf1[C_d(w)>0]$. Let $B(w,c)=\sum_{d:y_d=c}I_d(w)$. Then:

$$
P(w\mid c)=\frac{B(w,c)+\alpha}{\sum_{v\in V}B(v,c)+\alpha|V|},\qquad
\operatorname{score}(c,d)=\log P(c)+\sum_{w\in V}I_d(w)\log P(w\mid c).
$$

Each present word contributes once; absent words contribute nothing. Binarize consistently during training and prediction.

**Bernoulli NB** models every vocabulary word as a binary variable, including absence. Let $q_{wc}$ be the probability that a class-$c$ document contains $w$. Estimate it as $B(w,c)/N_c$, or with symmetric smoothing as $(B(w,c)+\alpha)/(N_c+2\alpha)$: each variable has two outcomes. The likelihood is:

$$
P(d\mid c)=\prod_{w\in V}q_{wc}^{I_d(w)}(1-q_{wc})^{1-I_d(w)}.
$$

Its log score includes both $I_d(w)\log q_{wc}$ and $(1-I_d(w))\log(1-q_{wc})$. The $q_{wc}$ values need not sum to one across words: each word has its own presence/absence distribution.

For vocabulary viagra, lottery, meeting, lunch, suppose presence probabilities are $(.80,.70,.10,.05)$ for spam and $(.01,.05,.60,.50)$ for ham. A document containing only meeting has:

$$
P(d\mid\text{spam})=.10(.20)(.30)(.95)=.0057,\qquad
P(d\mid\text{ham})=.60(.99)(.95)(.50)=.28215.
$$

Absent spam-associated words supply evidence for ham; also include class priors to predict. Binarizing multinomial inputs does **not** produce a Bernoulli likelihood. Eighty repetitions in one email count as one document containing a word, not eighty such documents.

## Vocabulary, assumptions, and limits

Tokenization determines the vocabulary, so keep it consistent across training and prediction. Modern systems may use subword tokens. A simple fixed-vocabulary NB policy ignores test tokens outside $V$. Smoothing instead handles a word inside $V$ that is missing from a particular class. An explicit unknown-token alternative requires learning its behavior consistently during training.

Stop-word removal is not automatically helpful: common words can carry task-specific evidence. Weighting features, for example with tf-idf, downweights frequent terms but changes the representation; evaluate the result rather than assuming it improves NB.

NB is inexpensive and useful with small datasets. If its independence model and probability estimates match the data, its posterior gives the Bayes-optimal class decision. Language violates its assumptions: correlated features can cause repeated counting of evidence, and class priors can disadvantage rare classes. Useful feature design still matters.

NB is generative: it models a prior and how documents are generated given a class. Logistic regression is discriminative: it models $P(c\mid d)$ directly and permits overlapping features without NB's independence assumption. A more complex model is not automatically better on every dataset.

## Evaluate the errors that matter

For a positive class, TP means predicted positive and actually positive; FP means predicted positive but actually negative; FN means predicted negative but actually positive; TN means predicted negative and actually negative. Label confusion-matrix axes before reading these counts.

$$
\mathrm{Accuracy}=\frac{TP+TN}{N},\qquad
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN}.
$$

Precision measures correctness among selected positives; recall measures recovery of actual positives. Accuracy can hide minority-class errors. On the same set of 75 positive and 175 negative examples, two classifiers can have equal accuracy but different error tradeoffs:

| Classifier | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 70 | 45 | 5 | 130 | .80 | .609 | .933 | .737 |
| B | 50 | 25 | 25 | 150 | .80 | .667 | .667 | .667 |

F1 is the harmonic mean of precision and recall:

$$
F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}.
$$

It requires both measures to be reasonably high. More generally:

$$
F_\beta=\frac{(1+\beta^2)PR}{\beta^2P+R}
=\frac{(1+\beta^2)TP}{(1+\beta^2)TP+\beta^2FN+FP}.
$$

$\beta>1$ emphasizes recall; $\beta<1$ emphasizes precision. When denominators are zero, state the metric convention used rather than treating an undefined ratio as an ordinary number.

### Macro and micro averages

For multiclass evaluation, treat each class as positive in turn. A macro average computes the metric per class, then takes an unweighted mean. Macro F1 is $K^{-1}\sum_cF_{1,c}$, **not** F1 calculated from macro precision and recall.

A micro average sums class-level TP, FP, and FN first, then computes the metric from those totals. Common classes contribute more counts. For three classes with $(TP,FP)=(8,11),(60,55),(200,33)$:

$$
P_{\mathrm{macro}}=\frac13\left(\frac8{19}+\frac{60}{115}+\frac{200}{233}\right)\approx.600,\qquad
P_{\mathrm{micro}}=\frac{268}{268+99}\approx.730.
$$

Macro averaging exposes weak rare-class performance that pooling can obscure. In single-label multiclass classification evaluated over all classes, micro precision, recall, and F1 equal accuracy: each wrong prediction produces one FP and one FN.

### Thresholds and data splits

Predict positive when posterior odds exceed a threshold:

$$
\frac{P(+\mid d)}{P(-\mid d)}>T.
$$

For $T>0$, this is equivalent to $P(+\mid d)>T/(1+T)$; $T=1$ gives the usual .5 threshold. Increasing the threshold selects fewer positives, generally trading recall for precision. Sweep thresholds to obtain a precision-recall curve, then choose a task-appropriate threshold, such as maximum validation F1 or a required precision.

Fit parameters, vocabulary, and preprocessing statistics on training data. Tune smoothing and thresholds using validation data; reserve test data for evaluation after these decisions. Repeatedly selecting settings using test scores leaks information into model selection even when test examples never participate in parameter updates.
