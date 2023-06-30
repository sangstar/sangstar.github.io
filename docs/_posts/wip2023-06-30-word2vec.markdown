---
layout: post
title:  "Words as vectors"
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

Naive Bayes or logistic regression can be perfectly suitable tools for things like text classification, but the way it handles words is antiquated. Modern methods eschew Bayesian statistics