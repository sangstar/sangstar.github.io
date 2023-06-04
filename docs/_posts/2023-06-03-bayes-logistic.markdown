---
layout: post
title:  "Generative models, discriminative models, and what everyone gets wrong about naive Bayes"
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

$$\hat c = \underset{c \in C}{\text{argmax}} \ P(c|d)$$

The naive Bayes classifier then applies  Bayes' rule in statistics:

$$P(x|y) = \frac{P(y|x)P(x)}{P(y)}$$

such that it views a classifier's task as

$$\hat c = \underset{c \in C}{\text{argmax}} \ \frac{P(d|c)P(d)}{P(c)}$$

Since we are computing the equation above for each possible class, $$P(d)$$ is a constant throughout all our calculations and can be discarded as it has no bearing on the result. Therefore we can express it as:

$$\hat c = \underset{c \in C}{\text{argmax}} \ P(d|c)P(d)$$

Naive Bayes is called a *generative* model. $$P(d \vert c)P(d)$$ can be expressed as the joint probability distribution $$P(c,d)$$ due to the conditional probability density function

$$P(y|x) = \frac{P(x,y)}{P(x)}$$

for $$p(x) > 0$$. As such, naive Bayes actually attempts to model the joint probability $$P(c,d)$$ rather than $$P(c \vert d)$$ directly. Models of this form are called generative in that they learn the joint probability distribution $$P(c, d)$$ and can thereby theoretically generate documents by fixing a $$c$$ and sampling documents in $$P(d \vert c)$$. 

Noting that a document $$d$$ is a vector of features (tokens) we can express our original classifier as 

$$ \hat c = \underset{c \in C}{\text{argmax}} \ P(w_1, w_2, w_3, \ \ldots \ , w_n \vert c) \ P(c)$$

Keep in mind that we have reached this equation with no loss of generality -- this is true for all classifiers. However, the whole point of naive Bayes is arriving at the above equation and applying the **naive Bayes assumption**: that the probabilities given the class $$c$$ for $$P(w_i \vert c)$$ are 'naively' considered *independent* and are therefore able to be multiplied. The naive Bayes assumption is therefore:

$$P(w_1, w_2, ..., w_n \vert c) = P(w_1 \vert c) \cdot P(w_2 \vert c) \ \cdot \ \ldots \  \cdot P(w_{n-1} \vert c) \cdot P(w_n \vert c)$$

Thus our naive Bayes classifier is:

$$\hat c = c_{NB} = \underset{c \in C}{\text{argmax}} \ P(c) \ \underset{i}{\prod} P(w_i \vert c)$$

where $$i$$ are the word positions in the document $$w$$, so basically just making sure the product is taking into account word order. The calculation is typically done in log space to avoid underflow due to a product of probabilities, so we rewrite as:

$$c_{NB} = \underset{c \in C}{\text{argmax}} \ \log{P(c)} \ \underset{i}{\sum} \log{P(w_i \vert c)}$$

We've now turned out classifier into argmaxxing a sum of linear sum of features. Keep that in mind for later. I'm not going to get into how naive Bayes is trained, but it uses maximum likelihood estimate to calculate $$P(w_i \vert c)$$. 

Naive Bayes' assumption is generally a rather poor one, but it tends to do better than one might think given its "naivety". Since it works using maximum likelihood estimation it scales wonderfully, handling both small and large training datasets well, and excels in higher dimensional data due to training in linear time and can ignore feature dependencies irrelevant to the classification tasks (as it ignores all of them). It also does surprisingly well in NLP since for text data, even though Naive Byaes disregards dependencies between words, often just the occurences of words in a document is enough information to make decent predictions. 

## Linear regression 
Let's return to the original classification model from before.

$$\hat c = \underset{c \in C}{\text{argmax}} \ P(c|d) = P(d|c)P(d)$$

A linear regression model is a *discriminative* one. That is to say that instead of applying Bayes' rule, we aim to directly calculate $$P(c \vert d)$$. This is actually a key distinction. While a generative model can "understand" the classes in a sense by being able to generate examples belonging to them, a discriminative model cannot. It is purely concerned with separating classes and isn't concerned with what characterizes them. This is because generative models force themselves to model the joint probability distribution $$P(c,d))$$ to inform their predictions, while discriminative is *only* concerned with finding $$P(c \vert d)$$. If a logistic regression classifier was trained to classify horses or humans, it won't necessarily be able to tell you that humans have five fingers -- just that they don't have hooves. A generative model meanwhile could analogously "draw" a human. 

Anyway, the way logistic regression computes $$P(c \vert d)$$ is completely different to naive Bayes and more inline with "conventional" machine learning. Traditionally, it seeks to solve the probability of a positive class given feature vector $$x$$ $$P(y = 1 \vert x)$$ and the probability of a negative class $$P(y = 0 \vert x)$$ It learns from a training set a vector of weights and a bias term. Each weight $$w_i$$, well, *weighs* $$x_i$$. A scalar value whose magnitude indicates its "importance" when making a classification prediction, and its parity representing its sway towards the positive or negative class. The bias term is tacked on to the end as an intercept. Each feature $$x_i$$ is weighed by $$w_i$$ and summed, then offset by bias $$b$$:

$$z = \sum_{i=1}^n w_i x_i + b$$

The sum is definitionally a dot product so we can succinctly express this as

$$ z = \mathbf{w}  \cdot  \mathbf{x} + b$$

What we have so far is not a probability. It is not bounded between $$0$$ and $$1$$. Weights are real-valued and have no implicit restriction on their magnitude, so $$z$$ has unbound range. We've actually just derived *linear* regression. To output a probability, we need to pass this output through some mapping that constricts it between $$0$$ and $$1$$. Sigmoids are perfect for this. Let's wrap $$z$$ up in a sigmoid:

$$\sigma(z) = \frac{1}{1+\exp{(-z)}}$$

<p align="center">
  <img width="auto" height="auto" src="/assets/logistic.jpg">
</p>

This is actually more specifically a logistic function, a specific kind of sigmoid (hence *logistic regression*). Sigmoids are amazingly ubiquitous in ML, and its appearance here gives a few good examples. For one, it obviously has the desired range between $$0$$ and $$1$$ and this prevents outliers from having an undue influence as flattening occurs near the edges. It's also differentiable, which is always nice for models that fit using gradient descent. If we decide we want our sigmoid to model the probability that $$y = 1$$, we can state:

$$P(y=1) = \sigma(\mathbf{w}  \cdot  \mathbf{x} + b)$$

Since our only other outcome is $$y=0$$, we require $$P(y=1) + P(y=0) = 1$$. 

Hence,

$$P(y=0) = 1 - P(y=1) $$

$$ = 1 - \sigma(\mathbf{w}  \cdot  \mathbf{x} + b)$$

$$ = 1 - \frac{1}{1+\exp{(-(\mathbf{w}  \cdot  \mathbf{x} + b))}}$$

$$ = \frac{\exp{(-(\mathbf{w}  \cdot  \mathbf{x} + b))}}{1+\exp{(-(\mathbf{w}  \cdot  \mathbf{x} + b))}}$$

Keep in mind, logistic regression in NLP *does not require* a specific structure for the features -- it doesn't have to be a bag of words or word embeddings or whatever. Any property from the input can be a feature. I've already written an article talking about decision boundaries in logistic regression, but I want to stress that this existence of a decision boundary to make predictions is what makes this model discriminative, and its linearity is what makes it a linear classifier -- that and the fact that we use a weighted (but linear) sum of features to make a prediction. Naive Bayes does this too, remember? However, the topic of whether naive Bayes is a linear classifier or not is actually a bit complicated. 

$$c_{NB} = \underset{c \in C}{\text{argmax}} \ \log{P(c)} \ \underset{i}{\sum} \log{P(w_i \vert c)}$$


## An aside: a common misconception about naive Bayes as a linear classifier
If you Google whether Naive Bayes is a linear classifier, lots of stuff you'll read will say that it is, yet at the same time looking at images of naive Bayes will show non-linear decision boundaries. What gives? Hopefully this will clear things up.

Formally speaking, a linear classifier is a model that makes a classification decision using a linear combination of its features and some scalars (not arbitrarily considering calars would be to imply that all scalars are always equal to unity):

$$y = f \bigl(\sum_j w_j x_j \bigr)$$

where $$f$$ is some *threshold function* that we want to scale things from $$0$$ to $$1$$ for a probability and help us to establish a decision boundary. Recall that for logistic regression we literally derived 

$$P(y=1) = \sigma(\mathbf{w}  \cdot  \mathbf{x} + b)$$

Let's start with Bayes' rule:

$$P(c \vert d) = \frac{P(d|c)P(d)}{P(c)}$$

Now applying the law of total probability (which we can because we have distinct events of $$y=1$$ or $$y=0$$) to $$P(c)$$ we can rewrite this as

$$P(c) = P(x \vert c = 1)P(c = 1) + P(x \vert c = 0) P(c=0)$$

Hence we have 

$$P(c \vert d) = \frac{P(d|c)P(d)}{P(x \vert c = 1)P(c = 1) + P(x \vert c = 0) P(c=0)}$$

$$ = \frac{1}{1+\frac{P(x \vert c = 0) P(c=0)}}{P(x \vert c = 1)P(c = 1)}}$$

$$ = \frac{1}{1+\exp{-\log{\frac{P(x \vert c = 1) P(c=1)}}{P(x \vert c = 0)P(c = 0)}}} $$

## References

Lucas (https://stats.stackexchange.com/users/7733/lucas), How is Naive Bayes a Linear Classifier?, URL (version: 2015-03-18): https://stats.stackexchange.com/q/142258

Jurafsky, D., & Martin, J. H. (2019). Naive Bayes, Text Classification and Sentiment. In Speech and Language Processing (3rd ed., Chapter 4). Prentice Hall.