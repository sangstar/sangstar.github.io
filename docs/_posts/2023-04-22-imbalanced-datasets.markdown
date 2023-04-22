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

Accuracy as a metric is rife throughout machine learning libraries like Tensorflow, where validation accuracy is a common early stopping criterion for training. However, I personally hate accuracy as a metric, and relying on it is not only bad practice, but potentially costly. 

## Where accuracy goes wrong

Suppose a bank wants to create a classifier than determines whether bank transcations are fraudulent (`1`) or not (`0`). Suppose someone is tasked with creating this model, and creates a feed-forward network with inputs, two hidden layers with something like 16 units and 12 units each and RELU activations. That's a pretty boilerplate feed-forward. Suppose that when they train the model, they rely on validation accuracy as a stopping criterion. 

Suppose they start training, and all of a sudden the model is trained quickly and in a few epochs, with validation accuracy well over 90%. *This should be an expected result if you looked at the target distribution*, and you would probably *not* consider this model well-equipped to handle its use case if you were to look at its predictions on the test set. The model would be performing 3 matrix multiplication operations, activation function computations on 29 nodes, and yet I can create my own predictor that will likely do just as good a job in one line of code: `def predictor(x): return 0`

If the engineer were to look at his data, if the data is representative of a typical bank transactions dataset, fraudulent transactions would be horribly under-represented. This is an example of an imbalanced dataset, which accuracy is woefully equipped to handling. An imbalanced dataset is any dataset where the class distribution is not flat.