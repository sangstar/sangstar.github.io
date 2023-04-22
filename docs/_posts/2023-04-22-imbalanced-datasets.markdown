---
layout: post
title:  "Imbalanced datasets and the dangers of over and undersampling"
date:   2022-11-18 10:15
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


A common pitfall a beginner machine learning engineer can find themselves in is putting faith in the metric "accuracy". In classification tasks, accuracy is defined as:

$$ \frac{P_T + N_T}{P + N}$$

Where $$P_T$$ and $$N_T$$ represent the true and false positive results respectively made by a predictor, 