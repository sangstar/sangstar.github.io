---
layout: post
title:  "Preprocessing"
date:   2023-02-25 8:58
categories: ml
usemathjax: true
---

<!-- for mathjax support -->
{% if page.usemathjax %}
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
    });
  </script>
  <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
{% endif %}


In my view, most of work done in ML is preparing for training. Aggregating appropriate data, analyzing it, deciding if it's sufficient, and then preprocessing it in my view is the majority of the work of the machine learning engineer. Preprocessing is the preparation of data for maximal performance during training. It is done in both ML and NLP, in that it's done regardless of whether your data is numeric, categoric, or text. 

## Data scaling
# Standardization
A multilayer perceptron (MLP) is what you'll see when you google "neural network". MLP's are probably the single biggest thing that got me interested in ML. It applies non-linear (crucially) transformations to find a linear decision boundary in a latent space that becomes a non-linear boundary in the input space. That sounds heavy, but visually it's gorgeous. Check out this [fabulous article](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) by Christopher Olah or [this youtube video](https://www.youtube.com/watch?v=k-Ann9GIbP4) to get a better idea of what I'm talking about. 

In a MLP, hidden units mathematically represent hyperplanes that attempt to orient themselves orthogonal to a decision boundary. Since bias terms that are too large tend to cause saturation, the hyperplanes tend to sit fairly close to the origin. If the data cloud is therefore not closely spaced to the origin, the hyperplanes could not cut through any of the data cloud, especially with small variance, which amounts to very poor initialization. Very poor initialization tends to lead to local minima, so we want to avoid this by trying to transform our input data so that the hyperplanes have a better chance of passing through it. This can be achieved by setting the mean of all datapoints to 0 and setting their standard deviation to 1, which is called standardization.

So while standardizing input data for MLP allows for much better initialization, it's far more crucial in unsupervised learning algorithms, where if one of your columns has very high variance and another very low variance, the low variance column will have virtually no impact if using something like $$k$$-means with euclidean distance. This problem can easily and will often arise. If I'm training a model that has people's heights in one column and the money they spent on buying their house on the other, the latter will have a much, much higher variance and a clustering algorithm will basically ignore the heights column outright and unjustifiably. 

# Normalization
Models like MLPs actually place more importance on higher numbers than lower numbers *always and without context*, unlike humans. If a predicted value for a housing price is 5% off the true value, this error is far larger than someone's height in feet being 5% off, yet the loss function typically only cares about the displacement between true and predicted. These huge error calculations can at best dominate the average gradient of batch or at worst lead to exploding gradients. We're usually more concerned with relative scales. In order to account for this, we like to normalize our data so that no columns are given unfair importance. A common way to do this is with *min-max scaling*, although there are a couple ways. The transformation is represented with the equation below:

$$x' = \frac{x-x_{min}}{x_{max}-x_{min}}$$

It's usually never a bad idea to do this with numeric data, unless you have two columns where you think preserving the different magnitudes is important. 

## NLP preprocessing
NLP is probably one of the most data-hungry fields of machine learning, yet has a serious dearth of quality labeled data. This is one of the reasons why LLM's are so popular. It can be often hard to justify building your own model from scratch as opposed to just fine-tuning a LLM. 

This is due to the simple fact that all models suffer from the *curse of dimensionality*. As the number of columns of your dataset increases, so too does the size of your input vectors. Since in a neural network the input vector of length $$n$$ performs a matrix multiplication with a $$n$$ by $$m$$ weights matrix $$W_1$$, with $$m$$ being the number of neurons in the subsequent hidden layer, any subsequent column you add results in adding $$m$$ more weights which now need to be calibrated with more data. As well, higher dimensional datasets are almost always sparser, which must have its dimensionality reduced or be supported with, again, more data. 

Since most non-transformer (self-attention models have a clever way of dealing with this) NLP models treat each and every *unique word* in the entire *corpus* as a feature, the curse of dimensionality in NLP is a constant bane. Dimensionality reduction *and* more data is crucial. The dimensionality reduction used in NLP typically involves lowercasing all text, removal of special characters like commas, exclamation marks, apostrophes, full stops etc, removal of words typically assumed to not affect the sentiment of documents, and trying to standardize different verb tenses. The lowercasing and removal of special characters hopefully is self-explanatory, but I'll touch on stopwords and stemming/lemmatizing. 

# Stopwords
Words such as "and", "the", "a", and "is" are often removed from documents because they are thought to usually not affect the semantic meaning of the sentence, such that if you removed them, you'd still understand what's being said. There is no universally agreed upon list of stopwords to use, and depending on what Python library you use, you'll probably be cutting different ones and leaving others that you wouldn't have using another library. That should be approached with a bit of caution.

# Stemming and lemmatizing
Lemmatizing removes verb tenses and plurality from words, so that "change", "changes", "changed", "changing" etc all fall under the same feature "change". Stemming goes a step further and removes the word stem altogether.

# A thing to keep in mind..
Removing stopwords and stemming/lemmatizing I feel is done quite liberally and without any consideration as to whether it would actually improve performance. The golden rule is "try it with and without and see which performs better", but it is definitely not always the case that removing these words is worth it. Each word you remove usually does the affect the interpretability of your documents, and if you do cleaning too liberally you could end up confusing your model as interpretability continues to deteoriate. Getting more data on the other hand is a much less risky yet effective solution, albeit not always practical. Lowercasing and removing special characters almost never hurt, unless your dataset is filled with proper nouns that are otherwise words in your target language. 

I will also mention that if you're fine-tuning a BERT model, you shouldn't stem, lemmatize or remove stopwords. LLM's are typically trained on stopwords and unlemmatized and unstemmed words, so you gain nothing by removing them. 

## References

Sarle, W. S. (n.d.). Comp.ai.neural-Nets FAQ, part 2 of 7: Learning. faqs.org. Retrieved February 26, 2023, from http://www.faqs.org/faqs/ai-faq/neural-nets/part2/ 

Olah, C. (n.d.). Neural networks, manifolds, and topology. Neural Networks, Manifolds, and Topology -- colah's blog. Retrieved February 26, 2023, from https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ 