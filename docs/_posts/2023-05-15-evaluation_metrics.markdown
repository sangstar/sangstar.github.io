---
layout: post
title:  "Evaluation metrics in NLP"
date:   2023-05-15 10:15
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

I've written about some important things when it comes to evaluation in previous articles, but I wanted to dedicate one solely to it as it's probably the most important area of knowledge to be proficient in when using ML to solve problems in the real world. Without clear ways to judge the performance of a model and its performance against others designed to tackle the same problem, your model will (or at least should) not make it out of a dev environment. I will be covering a way to do both. 


## Precision and Recall

## Confusion matrices and F-measure

## Statistical Significance
When trying to work out if model $$A$$ is superior to model $$B$$, comparing them on one test set is bad practice and would be unacceptable evidence in most bodies of scientific literature. You will need to enter the domain of statistical hypothesis testing. 

Suppose we want to compare the model performances of model $$A$$, a recurrent neural network (RNN), to model $$B$$, a naive Bayes sentiment classifier on a test set $$T$$. 

Suppose you take the $$F_1$$ scores of both models on the test set  (let's say $$M(A, T)$$ for performance by $$A$$ on test set $$T$$ and $$M(B, T)$$ for performance by $$B$$ on test set $$T$$) and define the performance difference as:

$$
\delta(T) = M(A,T) - M(B,T)
$$

This performance difference is known as an *effect size*. Suppose $$\delta(T) = 0.2$$. According statistical hypothesis testing, this *does not imply* that $$A$$ is a superior model to $$B$$. Can we say for sure that $$A$$ is a better model than $$B$$? 


## References
Jurafsky, D., & Martin, J. H. (2019). Naive Bayes, Text Classification and Sentiment. In Speech and Language Processing (3rd ed., Chapter 4). Prentice Hall.
