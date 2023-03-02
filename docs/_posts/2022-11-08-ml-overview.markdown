---
layout: post
title:  "The super basics of model training"
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

This guide is designed to be a (very) brief and light introduction to machine learning that dabbles on some of the mathematical underpinnings of prediction and training. I'm planning on writing a few articles on some machine learning basics for the future.

## Types of Problems
The two main tasks of ML have to do with *regression* and *classification*. Regression is what you'd work with to predict housing prices, and classification is what you'd work with to predict handwritten digits. In the case of perceptrons, the output neuron is singular for regression, as the output is some real number, and a vector for classification (giving a usually softmaxed or sigmoidal real-valued score for each class). 

## What is the goal of ML?
Despite the enormous number of use cases for machine learning, it all boils down to one concept. Given a set of datapoints $$\{x_i\}_{i=1}^{k}$$ let there exist some function $$\hat f: x \to y$$ such that 

$$\hat f \left(\{x_i\}_{i=1}^{k}\right) = \{y_i\}_{i=1}^{k}$$

where $$x_i$$ is often called an *input* or a *feature*, and $$y_i$$ is often called an *output* or *target*. In the case of neural networks, by the universal approximatin theorem, there will **always** exist some neural network $$\bar f$$ that can be constructed to approximate $$\hat f$$, such that

$$\bar f \left(\{x_i\}_{i=1}^{k}\right) = \{\bar y_i\}_{i=1}^{k}$$

where $$\bar y_i$$ is called a *prediction*. In order to find our $$\bar f$$, we need to minimize a *loss function* with *backpropagation*. 

# Loss functions
A loss function is a function that helps us to uncover $$\bar f$$ by ideally finding its global minimum. There are many loss functions out there, but a common one for estimating real numbers (regression) is called the *residual sum of squares*:

$$\mathcal{L} = \sum_{i=1}^n \left(y_i - f(x_i)\right)^2$$

where $$f$$ is a candidate predictor. $$f$$ will depend on all weights and biases between all gaps in neuronal layers. Therefore, so will the cost function. In order to minimize the cost function, you'll naturally need to look for trivial $$\nabla \mathcal{L}$$. This is known as gradient descent, and will inform you on how to adjust your weights and biases.

Whether we choose to update the weights after one iteration or after going through all datapoints $$\{x_i\}_{i=1}^{k}$$ (one epoch) or after a batch of datapoints is a question of whether you want to use stochastic or batch or minibatch gradient descent. Stochastic gradient descent will look something like this:

$$ w:= w - \gamma \nabla \mathcal{L_i}(w)$$

where we choose to refer to our constant term $$\gamma$$ as the *learning rate* (although big nerds like me like to make the learning rate adaptive and therefore not constant to allow us to be fast without leaping over minima -- see [Adam](https://arxiv.org/abs/1412.6980)). I personally don't like stochastic gradient descent as it is very sensitive to the initial data points you feed it and as a result can be a bit finnicky. It's typically only preferred in cases of really high-dimensional data to reduce computational burden. Because there is only one datapoint used per weight update, perturbations aren't averaged out, which can cause somewhat suspect gradients. This can have the somewhat surprising ability, however, of potentially avoiding a local minimum in favor of a global minimum, because of the inherent unpredictability of the loss's trajectory. Minibatch gradient descent updates weights with the mean gradient of $$N$$-sample minibatch:

$$ w:= w - \frac{\gamma}{n} \sum_{i+1}^n \nabla \mathcal{L_i}(w)$$

which isn't as effected by stochastic gradient descent, as the bigger the batch size the better it mitigates the risk of unrepresentative samples through averaging. 

I like to take a few pointers from the oft-described analogy for gradient descent of a hiker trying to find the top of a mountain in a misty forest to describe averaging gradients. Imagine a very misty, alien planet with huge craters that look like mountains that go into the earth, rather than jutting from it, and there's a lot of astronauts trying to find the largest crater. The fog is so dense that you can only see so far in front of you, but enough to have a general idea of where to move to get to a crater. Each astronaut can have their own idea of where the direction of the crater is from their vantage point, but if you were wanting to get there, wouldn't it make sense to get a collective idea from all of their measurements? You're less reliant on a specific astronaut you choose being right. 

Anyway, as the loss function minimizes the hope is that your $$f$$ converges to something approximating $$\bar f$$. It will often not, and there are many reasons as to why this could be the case which I won't get into right now. I'm also going to neglect talking about activation functions or some other stuff specific to neural nets, other than to say that activation functions allow you to approximate non-linear $$\bar f$$, as the output of a neuron is classically linear and linear functions strictly cannot approximate non-linear functions (piecewise linear functions are non-linear), while non-linear functions can approximate linearity (any physicist will give you the famous example that allows you to solve for the equation of motion for a simple pendulum $$\sin{\theta} \approx \theta$$ for small $$\theta$$). Weight matrices in essence apply linear transformations to your data cloud that are then distorted further by non-linear functions (like a sigmoid or hyperbolic tangent), and without the latter your decision boundary could never be non-linear, in which case your neural network essentially becomes a dressed-up linear model. I will note that you aren't guaranteed to "find" $$\hat f$$. It exists for your problem if there is an actual relationship between input and output, but you may not have the right hyperparameters or the right data or a robust (enough model parameters to learn the specific patterns) enough model to find it. 

# The training process
In order to train a model, say, a perceptron, you start with a dataset and with any luck some labeled data $$
D = \{x_i,y_i\}_{i=1}^{k}$$. You then want to see how the model does on data it hasn't "seen" before (some values of $$y_i$$ that *haven't* influenced the loss function) to assess how well the trained model does on the dataset. This involves splitting $$D$$ into *training* and *test* sets. It would also be smart to split the data into a *validation* set, but I'm going to neglect going into cross-validation right now but you can read about it in the second part of this series. With your training data, you update the weights of your neural net (or the parameters of your model, generally) by calculating all the gradients until you've reached trained it on a set number of epochs, or when early stopping criteria are met if you have any. After this is done, your model is considered trained, and you can see how the model's predictions $$\{\bar y_i\}_{i=1}^{k}$$ compare to $$\{y_i\}_{i=1}^{k}$$ in your test set. If you're doing classification with more than one class, please don't use accuracy.



