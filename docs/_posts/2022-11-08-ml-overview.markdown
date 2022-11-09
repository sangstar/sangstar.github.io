---
layout: post
title:  "Machine learning for the mathematician 1"
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

This guide is designed to be a (very) brief and light introduction to machine learning that caters a bit to the maths. Some knowledge of multivariate calculus and linear maths might be useful to understand some of the concepts here.

# What is the goal of ML?
Despite the enormous number of use cases for machine learning, it all boils down to one concept. Given a set of datapoints $$\{x_i\}_{i=1}^{k}$$ let there exist some function $$\hat f: x \to y$$ such that 

$$\hat f \left(\{x_i\}_{i=1}^{k}\right) = \{y_i\}_{i=1}^{k}$$

where $$x_i$$ is often called an *input* or a *feature*, and $$y_i$$ is often called a *target*. In the case of neural networks, by the universal approximatin theorem, there will **always** exist some neural network $$\bar f$$ that can be constructed to approximate $$\hat f$$, such that

$$\bar f \left(\{x_i\}_{i=1}^{k}\right) = \{\bar y_i\}_{i=1}^{k}$$

where $$\bar y_i$$ is called a *prediction*. In order to find our $$\bar f$$, we need to minimize a *loss function*.

# Loss functions
A loss function is a function that helps us to uncover $$\bar f$$ by ideally finding its global minimum. A machine learning model like a neural network 
