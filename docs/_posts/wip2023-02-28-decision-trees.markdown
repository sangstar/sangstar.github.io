---
layout: post
title:  "Decision trees and ensembles"
date:   2023-02-25 8:58
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

Decision trees are unique when it comes to machine learning models for multiple reasons. For one, they're probably by far, when made fairly simply, one of the easiest models to interpret, as they're prevalent in many fields outside ML. They also are a little unique when it comes to most models in that outside gradient boosting classifiers they don't use traditional loss. They're also a joy to learn, and learning a simple decision tree allows you to understand the building blocks of ensembling. In that vein, I'll start with covering what a decision tree is from a machine learning perspective.

## The decision tree
A decision tree's goal is to work top-down, sequentially applying a rule that "best" splits the data at each step. What qualifies as "best" can vary, but it generally is considered a good split when the two resulting subsets have a high target variable homogeneity. This process of partitioning subsets over and over, called recursive partitioning, continues until a subset has complete target variable homogenity, or if splitting is stopped early due to stopping criterion such as not exceeding the minimum samples per leaf, not exceeding some minimum impurity reduction, not being allowed to exceed a maximum depth, or no improved accuracy on a validation set by splitting further. I'll get into more of what these concepts mean in the article. 

## Impurity and information gain



## Ensemble methods

# Random Forest

# Gradient-boosting classifier

## When should I use decision trees or ensemble methods?


## References

Sarle, W. S. (n.d.). Comp.ai.neural-Nets FAQ, part 2 of 7: Learning. faqs.org. Retrieved February 26, 2023, from http://www.faqs.org/faqs/ai-faq/neural-nets/part2/ 

Olah, C. (n.d.). Neural networks, manifolds, and topology. Neural Networks, Manifolds, and Topology -- colah's blog. Retrieved February 26, 2023, from https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ 