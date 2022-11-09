---
layout: post
title:  "A crash course on ML: the super basics"
date:   2022-11-08 10:15
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

This guide is designed to be a (very) brief and light introduction to machine learning that caters a bit to the maths. Some knowledge of multivariate calculus and linear maths might be useful to understand some of the concepts here.

# What is the goal of ML?
Despite the enormous number of use cases for machine learning, it all boils down to one concept. Given a set of datapoints $$\{x_i\}_{i=1}^{k}$$ let there exist some function $$\hat f: x \to y$$ such that 

$$\hat f \left(\{x_i\}_{i=1}^{k}\right) = \{y_i\}_{i=1}^{k}$$

where $$x_i$$ is often called an *input* or a *feature*, and $$y_i$$ is often called a *target*. In the case of neural networks, by the universal approximatin theorem, there will **always** exist some neural network $$\bar f$$ that can be constructed to approximate $$\hat f$$, such that

$$\bar f \left(\{x_i\}_{i=1}^{k}\right) = \{\bar y_i\}_{i=1}^{k}$$

where $$\bar y_i$$ is called a *prediction*. In order to find our $$\bar f$$, we need to minimize a *loss function*.

# Loss functions
A loss function is a function that helps us to uncover $$\bar f$$ by ideally finding its global minimum. A common loss funciton is called the *residual sum of squares*:

$$\mathcal{L} = \sum_{i=1}^n \left(y_i - f(x_i))^2\right)$$

where $$f$$ is a candidate predictor. $$f$$ will depend on all weights and biases between all gaps in neuronal layers. Therefore, so will the cost function. In order to minimize the cost function, you'll naturally need to look for trivial $$\nabla \mathcal{L}$$. This is known as gradient descent, and will inform you on how to adjust you weights and biases. Recall from your vector calc class gradient descent can be represented like

$$ \Delta \mathbf{w_i} = - \gamma \nabla \mathcal{L}(\mathbf{w_i})$$

Where we choose to refer to our constant term as the *learning rate*. This is called backpropogation, or backprop. Whether we choose to update the weights after one iteration or after going through all datapoints $$\{x_i\}_{i=1}^{k}$$ (one epoch) is a question of whether you want to use stochastic or standard gradient descent. I personally don't like stochastic gradient descent as it is very sensitive to the order of datapoints you feed it and as a result can be a bit finnicky. Anyway, as the loss function minimizes the hope is that your $$f$$ converges to $$\bar f$$. It will often not, and there are many reasons as to why this could be the case, which I won't get into right now. I'm also going to neglect talking about activation functions or t

# This is just the surface..
A neural network is called a *linear classifier*, in that it involves some manipulation on a dot product of a feature vector with a weights vector (which, when wrapped up in an activation function represents the output of a neuron in a neural net). However, there are many other linear classifiers such as logistic regression (which funnily enough is a classifier), and support vector machines.

# Types of Problems
The two main tasks of ML have to do with *regression* and *classification*. Regression is what you'd work with to predict housing prices, and classification is what you'd work with to predict handwritten digits. In the case of perceptrons, the output neuron is singular for regression, as the output is some real number, and a vector for classification (giving a usually softmaxed or sigmoidal real-valued score for each class). 

