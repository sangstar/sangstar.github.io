---
layout: post
title:  "Imbalanced datasets and the dangers of over and undersampling"
date:   2023-04-22 10:15
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

Where $$P_T$$ and $$N_T$$ represent the true and false positive results respectively made by a predictor, whereas $$P$$ and $$N$$ represent the actual number of positive and negative cases in something like a test dataset. It's quite easy to compute when programming as long as your `==` operator is element-wise, and computing it between $$y$$ and $$\hat y$$: the true and predicted $$y$$ values. This will create an array of booleans whose average is the accuracy.  

Accuracy as a metric is rife throughout machine learning libraries like Tensorflow, where validation accuracy is a common early stopping criterion for training. 