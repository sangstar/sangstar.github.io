---
layout: post
title:  "Learning curves and cross-validation"
date:   2022-11-12 10:15
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

## Two very useful concepts!
Learning curves and cross-validation are integral things to incorporate when training your models. Learning curves help clue you in on how much labeled data you need and crucially helps inform you about whether your model is overfitting or underfitting. Cross-validation gives you a good idea of the *validation accuracy and loss* for each training epoch, which is a great metric to monitor for early stopping and is a necessary inclusion in your aforementioned learning curves in order to spot overfitting. I'll start with talking about cross-validation.

## Cross-validation
Cross-validation draws from a *validation set*. I mentioned it briefly in the first part of this series of ML posts. A validation set is a test set that is used *during* training, and typically takes the same share of the labeled data as the test set. Cross-validation aims to continually assess a model's ability to generalize as it's being trained. The most common form of cross-validation is the non-exhaustive $$k$$-fold cross-validation. 

# $$k$$-fold cross-validation
In $$k$$-fold cross-validation, a *fold* is the size of the training data to be partitioned into $$k$$ equally sized subsets. Of the $$k$$ folds, $$k-1$$ folds are used for training and one fold is reserved as a test set. From this partition, a model is trained on the training part and an accuracy score is computed on the test part of each fold. The scores from each of the folds are averaged and used as an evaluation metric. An image of this process, created by Wikipedia user [Gufosawa](https://commons.wikimedia.org/wiki/User:Gufosowa), can be found below:

<p align="center">
  <img width="auto" height="auto" src="/assets/kfold.jpg">
</p>

As you can see, the test data is a sliding window that glides over the whole dataset per epoch, such that the every datapoint is used once and only once as test data.

## Why is cross-validation a good idea?

From my article on [the basics of ML](https://sangstar.github.io/ml/2022/11/08/ml-overview.html) I talk about there being some ideal function $$\hat f$$ that we wish to approximate. One can imagine there being some set of approximating functions $$\{f_i\}^n$$ with varying effectiveness. The best approximator $$\bar f$$ in that set will have the lowest total variance and bias compared to its fellow members in the set. Cross-validation is good because it serves as a very useful evaluation metric to assist in finding $$\bar f$$.

# Bias and Variance

If you take a look at the loss function I made in my previously mentioned article, you'll note that its actually just an average of the squared difference between the true and predicted values of a dataset:

$$\mathcal{L} = \sum_{i=1}^n \left(y_i - f(x_i)^2\right)$$

which is often called the *mean squared error*, or MSE. The mean is called an expected value, and can actually be written more simply as this:

$$\mathcal{L} = \sum_{i=1}^n \left(y_i - f(x_i)^2\right) = \mathbb{E}\left[\left(y_i - f(x_i)\right)^2\right]$$

and there's actually a way of rewriting this equation:


$$\mathbb E\left[\left(y_i - f(x_i)\right)^2\right] = \left(\mathbb E\left[y_i - f(x_i)\right]\right)^2 +\mathbb E\left[\left(f(x_i) - \mathbb E\left[f(x_i)\right] \right)^2\right]$$

Of that new expression on the right-hand side, the first term is called the *square of the expected bias*, and the second term is called the variance of the estimator $$f$$.




## References

Henry (https://math.stackexchange.com/users/6460/henry), difference between bias vs variance, URL (version: 2020-05-10): https://math.stackexchange.com/q/3667818