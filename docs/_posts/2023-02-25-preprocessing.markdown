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


In my view, most of work done in ML is preparing for training. Aggregating appropriate data, analyzing it, deciding if it's sufficient, and then preprocessing it is the majority of the work of the machine learning engineer. Preprocessing is the preparation of data for maximal performance during training. It is done in both ML and NLP, in that it's done regardless of whether your data is numeric, categoric, or text. 

## Types of preprocessing and why it's done
# Standardization
In a multilayer perceptron (MLP), which is what you'll see when you google "neural network". MLP's are probably the single biggest thing that got me interested in ML. It applies non-linear transformations to find a linear decision boundary in a latent space that becomes a non-linear boundary in the input space. That sounds heavy, but visually it's gorgeous. Check out this [fabulous article](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) by Christopher Olah or [this youtube video](https://www.youtube.com/watch?v=k-Ann9GIbP4) to get a better idea of what I'm talking about. 

In an MLP, hidden units mathematically represent hyperplanes that attempt to orient themselves orthogonal to the decision boundary. Since bias terms that are too large tend to cause saturation, the hyperplanes tend to fairly close to the origin. If the data cloud is therefore not closely spaced to the origin, the hyperplanes could not cut through any of the data cloud, especially with small variance, which amounts to very poor initialization. Very poor initialization tends to lead to local minima, so we want to avoid this by trying to transform our input data so that the hyperplanes have a better chance of passing through it. This can be achieved by setting the mean of all datapoints to 0 and setting their standard deviation to 1.

So while standardizing input data for MLP allows for better initialization, it's far more crucial in unsupervised learning algorithms, where if one of your columns has very high variance and another very low variance, the low variance column will have virtually no impact if using something like $$k$$-means with euclidean distance. This problem can easily and will often arise. If I'm training a model that has people's heights in one column and the money they spent on buying their house on the other, the latter will have a much, much higher variance and a clustering algorithm will basically ignore the heights column unfairly. 

# Normalization
Models like MLPs actually place more importance on higher numbers than lower numbers *always and without context*, unlike humans. We're more concerned with relative scales, while lots of models deal with *absolutes*. In order to remove this bias, we like to normalize our data so that no columns are given unfair importance. A common way to do this is with *min-max scaling*. 