---
layout: post
title:  "Why can't regression models perform classification?"
date:   2022-11-24 10:15
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

You might've noticed that logistic *regression* is more often than not used as a classifier. Then why can't I use linear regression as a classifier? This is a question many of us have asked ourselves when first learning about the two models, and probably take for granted. But the answer is not as simple as it may seem, and is a great lesson in what makes a model a classifier in the first place. 

## What is a regressor? What is a classifier?

A regressor is a model that predicts a *quantity*, specifically some real number $$x \in \mathbf{R}$$. A classifier, on the other hand, essentially outputs a vector that is typically argmax'd to a sparse vector, with the $$1$$ positioned at the predicted class. They are the two standard approaches to supervised learning, and are well-known to any ML enthusiast. 

## What allows a regressor to be used as a classifier?

Quite simply, *a decision rule*. Logistic regression is a *regression algorithm*, plain and simple. It just happens to be a good classifier if you give it a decision rule like:

$$a \text{if} x >= 0.5 \text{else} b$$

