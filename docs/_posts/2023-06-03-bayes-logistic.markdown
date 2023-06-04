---
layout: post
title:  "Generative and Discriminant models: Naive Bayes and Logistic Regression"
date:   2023-06-03 10:42
categories: nlp
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

When learning about machine learning models, especially in the context of classification (and NLP), it is usually a pretty good idea to learn simpler models and then build up to more complex ones. A fairly common order is naive Bayes and then logistic regression. Logistic regression is a good precursor to more complex stuff like neural networks, as it captures all the weights optimization stuff while still being pretty interpretable. I've talked a bit about logistic regression in the past briefly (see my [article](https://sangstar.github.io/ml/2022/11/24/classifying-with-regression.html)) but haven't delved into it formally, and I've never touched on naive Bayes. You probably won't ever use them (especially naive Bayes) if you value performance above everything, but they're still incredibly important to learn foundationally. They're also different "philosophically", in that they represent two different approaches to classification; naive Bayes being a *generative* model, and logistic regression being a *discriminative* model. I'm going to talk about all of this in this article, starting with naive Bayes and the generative model.

## Naive Bayes and the goal of Classification
The naive Bayes model for classification in NLP is a good next thing to learn after starting with the $$n$$-gram model. The goal of any classifier $$\hat c$$ is to satisfy the following:

$$c = \underset{c \in C}{\text{argmax}} \ P(c|d)$$

The naive Bayes classifier then applies the Bayes rule in statistics:

$$P(x|y) = \frac{P(y|x)P(x)}{P(y)}$$

such that it views a classifier's task as

$$c = \underset{c \in C}{\text{argmax}} \ \frac{P(d|c)P(d)}{P(c)}$$

Since we are computing the equation above for each possible class, $$P(d)$$ is a constant throughout all our calculations and can be discarded as it has no bearing on the result. Therefore we can express it as:

$$c = \underset{c \in C}{\text{argmax}} \ P(d|c)P(d)$$

This is called a generative model. $$P(d|c)P(d)$$ can be expressed as the joint probability distribution $$P(c,d)$$ due to the conditional probability density function

$$p(y|x) = \frac{p(x,y)}{p(x)}$$

for $$p(x) > 0$$. As such, naive Bayes actually attempts to model the joint probability $$P(c,d)$$ rather than $$P(c|d)$$ directly. Models of this form are called generative in that they learn the joint probability distribution $$P(c, d)$$, which allows them to find $$P(c|d)$$ by for each $$c$$ generating documents $$d$$ to maximize $$P(d|c)P(c)$$.