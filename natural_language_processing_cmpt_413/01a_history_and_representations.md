# NLP history and representations

The core modeling choices are representation, composition, and prediction. NLP has progressed from hand-written rules to statistical models and learned representations.

## From rules to learned representations

Rule-based systems combine human-designed patterns with a representation of the task. A rule might associate a keyword such as “delicious” with positive sentiment or match a word ending in `ed` as a possible past-tense verb. Such patterns can be useful, but exceptions, context, and ambiguity make a comprehensive rule set difficult to maintain.

SHRDLU illustrated this approach in a restricted blocks world. It could interpret a request such as “Pick up a big red block” using language rules and knowledge of the scene, and ask for clarification when “the pyramid” did not identify a unique object. Success in a carefully bounded world did not remove the difficulty of processing unrestricted language.

Programming languages are designed with specified syntax and semantics. Natural languages evolve through use and transmit complex meanings that depend on context. A surface form can have several interpretations, so writing a rule for every possible expression does not scale.

Statistical NLP learns parameters from electronic corpora, often using hand-designed features. Its growth was supported by more data, more computation, and evaluation on held-out examples. Statistical machine translation progressed from word-level models to phrase-based models.

Neural NLP learns intermediate representations along with the prediction model. Representation learning reduces the need to specify every input feature manually; deep learning composes multiple layers of increasing abstraction. Neural translation and conversational systems demonstrate the gains from this approach. Large-scale supervision and computation remain major ingredients: parallel translation sentences provide clear paired targets, while other domains may lack enough data or even an agreed annotation target.

Reinforcement learning from human feedback (RLHF) appears as a later development in conversational systems: human feedback helps guide model behavior. This is a high-level historical example; the central lesson here is the progression from manually specified rules and features toward learned representations and feedback-driven training.

## Scale and sparsity

Language contains many phenomena, and text frequencies are heavy-tailed: a small set of patterns is common, while many words, senses, and combinations are rare. A larger corpus helps without ensuring that every relevant case has been observed.

Low-resource languages and specialized domains may have little usable training data even when other languages have enormous web corpora. Models must generalize beyond observed examples, and performance on a large general corpus does not establish performance in every language or domain.

## Represent meaning and translate between surface forms

A representation determines which information a computer can use. An **attributed representation** describes a concept with explicit properties, such as whether a cat is a mammal, has fur, and meows. The properties are interpretable, but someone must define them and supply their values.

One-hot vectors identify vocabulary items. Different words occupy different coordinate axes, so distinct words are orthogonal and the representation does not express their semantic similarity. Dense embeddings use a shared space of real-valued coordinates. Similar usage can support similar representations, allowing a learned model to share information between words.

The **Vauquois triangle** organizes translation by how much analysis separates source text from target text:

- **Direct translation:** map the source surface form toward the target surface form with little intermediate abstraction.
- **Transfer:** analyze the source into an intermediate structure, transfer it to a target-language structure, then generate target text.
- **Interlingua:** analyze the source into a language-independent meaning representation, then generate the target language from that representation.

Moving upward means performing deeper analysis before generation. A learned vector can serve as an intermediate representation, but its being a vector does not by itself prove that it contains complete meaning or is independent of language.

## Learn representations from context

Start with words as discrete tokens, associate each with a trainable vector, and learn through a prediction task. The context supplies information about the missing or upcoming token: “The cat sat on the _____” or “A bottle of _____ is on the table.” Training adjusts representations so that they help predict those tokens.

This provides supervision from text without manually writing semantic attributes for every word. Words with similar contexts can receive similar learned representations because that helps the prediction model.

## Compose representations and use them

A sentence requires combining representations of its words or characters. A bag of words ignores order; a bag of n-grams retains short ordered patterns but discards how those patterns are arranged. RNNs, CNNs, transformers, and graph neural networks offer different ways to compose sequences or structures.

A simple aggregation is a weighted average. For word vectors $e_{w_i}$ and nonnegative weights $a_i$ with a positive total,

$$s=\frac{\sum_{i=1}^T a_i e_{w_i}}{\sum_{i=1}^T a_i}.$$

Equal weights give the ordinary mean. This loses word order: “dog bites person” and “person bites dog” have the same mean despite their different meanings. Models that retain sequence or structural information can distinguish them.

The resulting representation supports a downstream prediction, such as spam classification, or decoding that generates text. It can also condition output in another modality, such as an image. Representation, composition, and prediction are connected design choices: useful information depends on the task the representation must support.
