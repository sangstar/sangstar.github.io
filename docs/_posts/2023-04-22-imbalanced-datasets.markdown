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

Suppose a bank wants to create a classifier than determines whether bank transcations are fraudulent (`1`) or not (`0`). Suppose someone is tasked with creating this model, and trains a support vector machine, or maybe a basic decision tree for interpretability. Suppose that when they train the model, they rely on validation accuracy as a stopping criterion. 

Suppose they start training, and all of a sudden the model is trained quickly and in a few epochs, with validation accuracy well over 90%. *This should be an expected result if you looked at the target distribution*, and you would probably *not* consider this model well-equipped to handle its use case if you were to look at its predictions on the test set. The model would be performing 3 matrix multiplication operations, activation function computations on 29 nodes, and yet I can create my own predictor that will likely do just as good a job in one line of code: `def predictor(x): return 0`

If the engineer were to look at his data, if the data is representative of a typical bank transactions dataset, fraudulent transactions would be horribly under-represented. This is an example of an imbalanced dataset, which accuracy is woefully equipped to handling. An imbalanced dataset is any dataset where the class distribution is not flat. A *balanced dataset* is below, with arbitary classes $$A$$ and $$B$$.

(image)

Whereas an imbalanced dataset is any deviation from this parity, such as if I reduced the prevalence of some classes at random like below:

(image)

The first image is fairly straight-forward to train, but unrealistic -- you'll probably find it is seldom the case that this parity will occur naturally. 

Imbalanced datasets are a messy subject. Once aspiring machine learning engineers learn about imbalanced datasets, they tend to invariably assume it is a problem that needs to be fixed. One of the main reasons for this writing this article is as a PSA for machine learning engineers. 

Ladies and gents: the issue isn't that there is class imbalance. The issue is that you're dealing with a cost-sensitive learning problem. That is to say that the misclassification costs lack parity. 

## Why models favor the majority class
Your model is probably ignoring the minority class because it is lacking the complexity to capture the patterns of the under-represented class and also is typically incentivized to favor the majority class when you pick an ill-equipped loss function. 

To explain my first part, suppose I'm using a support vector machine to separate class $$A$$ and $$B$$ with a hyperplane on the feature space. If I have a huge dearth of datapoints for class $$B$$, it's incredibly difficult to find a reliable hyperplane orientation to separate the classes. With more datapoints, the data cloud for $$B$$ will ideally become more visible, which will allow the model to have less trouble fitting its hyperplane.

Additionally, if I worked with another primitive model like a decision tree, which recursively partitions the feature space based on chosen feature values, too little data might cause the tree to not have enough information to distinguish the minority class $$B$$ from $$A$$.

This can be remedied to some extent with a more complex model like a neural network, thanks to its ability to fit non-linear decision boundaries quite well. Even still though, more data will greatly help. 

This may not be enough, however. Cross-entropy is the flagship loss function for classification tasks, and is defined for two probability distributions $$P$$ and $$Q$$ as:

$$H(P,Q) = - \sum_{i=1}^N p(x_i) \log{(q(x_i))}$$

where $$p(x_i)$$ is interpreted as the true distribution and $$q(x_i)$$ as the predicted distribution. 

I kind of like writing it in terms of the dot product

$$L = \mathbf{y} \cdot \log{(\hat \mathbf{y})}$$

where $$L$$ now represents cross-entropy as a loss function, $$\mathbf{y}$$ as the vector of values consisting of $$p(x_i)$$ and $$\hat \mathbf{y}$$ as the vector of values consisting of $$q(x_i)$$. 

Cross-entropy is super nice for classification because it's convex (which is obviously ideal for a loss function) and well-suited to backpropagation. The logarithm in its equation is also particularly handy, punishing incorrect classifications (due to its behavior $$x \to \infty$$) by blowing up if the probability of the correct class is low according to $$q(x_i)$$