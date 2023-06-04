---
layout: post
title:  "Generative and Discriminant models: Naive Bayes and Logistic Regression"
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

$$c = \underset{c \in C}{\text{argmax}} \ P(c|d)$$

The naive Bayes classifier then applies the Bayes rule in statistics:

$$P(x|y) = \frac{P(y|x)P(x)}{P(y)}$$

such that it views a classifier's task as

$$c = \underset{c \in C}{\text{argmax}} \ \frac{P(d|c)P(d)}{P(c)}$$

Since we are computing the equation above for each possible class, $$P(d)$$ is a constant throughout all our calculations and can be discarded as it has no bearing on the result. Therefore we can express it as:

$$c = \underset{c \in C}{\text{argmax}} \ P(d|c)P(d)$$

Naive Bayes is called a *generative* model. $$P(d \vert c)P(d)$$ can be expressed as the joint probability distribution $$P(c,d)$$ due to the conditional probability density function

$$p(y|x) = \frac{p(x,y)}{p(x)}$$

for $$p(x) > 0$$. As such, naive Bayes actually attempts to model the joint probability $$P(c,d)$$ rather than $$P(c \vert d)$$ directly. Models of this form are called generative in that they learn the joint probability distribution $$P(c, d)$$ and can thereby theoretically generate text by fixing a $$c$$ and sampling documents in $$P(d \vert c)$$. 

Noting that a document $$d$$ is a vector of features (tokens) we can express our original classifier as 

$$ \hat c = \underset{c \in C}{\text{argmax}} \ P(w_1, w_2, w_3, \ \ldots \ , w_n \vert c) \ P(c)$$

Keep in mind that we have reached this equation with no loss of generality -- this is true for all classifiers. However, the whole point of naive Bayes is arriving at the above equation and applying the **naive Bayes assumption**: that the probabilities given the class $$c$$ for $$P(w_i \vert c)$$ are 'naively' considered *independent* and are therefore able to be multiplied. The naive Bayes assumption is therefore:

$$P(w_1, w_2, ..., w_n \vert c) = P(w_1 \vert c) \cdot P(w_2 \vert c) \ \cdot \ \ldots \  \cdot P(w_{n-1} \vert c) \cdot P(w_n \vert c)$$

Thus our naive Bayes classifier is:

$$\hat c = c_{NB} = \underset{c \in C}{\text{argmax}} \ P(c) \ \underset{i}{\prod} P(w_i \vert c)$$

where $$i$$ are the word positions in the document $$w$$, so basically just making sure the product is taking into account word order. The calculation is typically done in log space to avoid underflow due to a product of probabilities, so we rewrite as:

$$c_{NB} = \underset{c \in C}{\text{argmax}} \ P(c) \ \underset{i}{\sum} P(w_i \vert c)$$

We've now turned out classifier into argmaxxing a sum of linear sum of features. Keep that in mind for later. I'm not going to get into how naive Bayes is trained, but it uses maximum likelihood estimate to calculate $$P(w_i \vert c)$$. 

Naive Bayes's assumption is generally a rather poor one, but it tends to do better than one might think given its "naivety". Since it works using maximum likelihood estimation it scales wonderfully, handling both small and large training datasets well, and excels in higher dimensional data due to training in linear time and can ignore feature dependencies irrelevant to the classification tasks (as it ignores all of them). It also does surprisingly well in NLP since for text data, even though Naive Byaes disregards dependencies between words, often just the occurences of words in a document is enough information to make decent predictions. 

## Linear regression 
Let's return to the original classification model from before.

$$c = \underset{c \in C}{\text{argmax}} \ P(c|d) = P(d|c)P(d)$$

A linear regression model is a *discriminative* one. That is to say that instead of applying Bayes's rule, we aim to directly calculate $$P(c \vert d)$$. This is actually a key distinction. While a generative model can "understand" the classes in a sense by being able to generate examples belonging to them, a discriminative model cannot. It is purely considered with separating classes and isn't concerned with what characterizes them. This is because generative models force themselves to model the joint probability distribution $$P(c,d))$$ to inform their predictions, while discriminative is *only* concerned with finding $$P(c \vert d)$$. If a logistic regression classifier was trained to predictor horse or humans, it won't be able to tell you that humans have five fingers -- just that they don't wear hooves.