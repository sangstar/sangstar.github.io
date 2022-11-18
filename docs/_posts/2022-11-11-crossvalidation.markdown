---
layout: post
title:  "Ramblings on ML 2: learning curves and cross-validation"
date:   2022-11-08 10:15
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
In $$k$$-fold cross-validatoin, a *fold* is the size of the training data to be partitioned into $$k$$ equally sized subsets. Of the $$k$$ folds, $$k-1$$ folds are used for training and one fold is reserved as a test set. From this partition, a model is trained on the training fold and an accuracy score is computed on the test fold




## References