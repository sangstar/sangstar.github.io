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

I've written about some important things when it comes to evaluation in previous articles, but I wanted to dedicate one solely to it as it's probably the most important area of knowledge to be proficient in when using ML to solve problems in the real world. Without clear ways to judge the performance of a model and its performance against others designed to tackle the same problem, your model will not (or at least should not) make it out of a dev environment. I will be covering a way to do both. 


## Precision and Recall

## Confusion matrices and F-measure

## Statistical Significance
When trying to work out if model $$A$$ is superior to model $$B$$, comparing them on one test set is bad practice and would be unacceptable evidence in most bodies of scientific literature. You will need to enter the domain of statistical hypothesis testing. 

Suppose we want to compare the model performances of model $$A$$, a recurrent neural network (RNN), to model $$B$$, a naive Bayes sentiment classifier on a test set $$T$$. 

Suppose you take the $$F_1$$ scores of both models on the test set  (let's say $$M(A, T)$$ for performance by $$A$$ on test set $$T$$ and $$M(B, T)$$ for performance by $$B$$ on test set $$T$$) and define the performance difference as:

$$
\delta(T) = M(A,T) - M(B,T)
$$

This performance difference is known as an *effect size*. Suppose $$\delta(T) = 0.2$$. According to statistical hypothesis testing, this *does not* allow us to state that $$A$$ is a superior model to $$B$$. That's because it's entirely possible that $$A$$ was "accidentally" better than $$B$$ on test set $$T$$, and that $$B$$ is actually no worse than $$A$$ or even better.  

In the realm of statistical inference, to make a claim that some agent $$A$$ outperforms $$B$$, we make two hypotheses: that $$A$$ is either as good or worse than $$B$$, or that $$A$$ is better than $$B$$. 

The first hypothesis is the one we assume is true, known as the null hypothesis. 

$$H_0 : \delta(T) \le 0$$

The goal to proving statistical significance in $$A$$'s superiority to $$B$$ is to find the empirical probability that we'd find our value of $$\delta(T)$$ or of one even greater if the null hypothesis is true. Basically we want to find the probability that we would see $$\delta(T)$$ or higher if $$A$$ is actually not better than $$B$$ with regard to some test statistic (in our case $$F_1$$). That is to say that we want to find for some arbitrary test set $$t$$:

$$P(\delta(t) \ge \delta(T) | H_0 \text{is true})$$

This probability is called a *p-value*. You might've seen the p-value before in studies and stuff, where it's usually set to something like $$0.05$$, which means there is a $$95\%$$ chance the null hypothesis is not true: that whatever $$B$$ is (in studies talking about medical interventions this is often called a placebo!) cannot be considered on par with or better than $$A$$ at some test statistic, like the difference in $$F_1$$ scores or the difference in amyloid deposition in the brain for an Alzheimer's drug. It's an *incredibly powerful* statistic.


## References
Jurafsky, D., & Martin, J. H. (2019). Naive Bayes, Text Classification and Sentiment. In Speech and Language Processing (3rd ed., Chapter 4). Prentice Hall.
