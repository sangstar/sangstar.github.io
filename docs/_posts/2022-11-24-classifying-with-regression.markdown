---
layout: post
title:  "Why can't regression models perform classification?"
date:   2022-11-24 10:15
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

You might've noticed that logistic *regression* is more often than not used as a classifier. Then why can't I use linear regression as a classifier? This is a question many of us have asked ourselves when first learning about the two models, and probably take for granted. But the answer is not as simple as it may seem, and is a great lesson in what makes a model a classifier in the first place. 

## What is a regressor? What is a classifier?

A regressor is a model that predicts a *quantity*, specifically some real number $$x \in \mathbf{R}$$. A classifier, on the other hand, essentially outputs a vector that is typically argmax'd to a sparse vector, with the $$1$$ positioned at the predicted class. They are the two standard approaches to supervised learning, and are well-known to any ML enthusiast. 

## What allows a regressor to be used as a classifier?

It's commonly thought that logistic regression works as classification due to using a decision rule. It seems to be a good classifier if you give it a decision rule like below.

$$a \ \ \text{if} \ \ y >= 0.5 \ \ \text{else} \ \ b$$


However, the *real* reason logistic regression is well-suited as a classifier is because its predictions are interpreted as the conditional probabilities $$P(y = 1 \mid x)$$. That way, you can model it as the likelihood that $$x_i$$ belong to class $$A$$. If it's low enough, you can *decide* it belongs to class $$B$$ instead if the probability is low. This leads people to adopting the decision rule above, but *you can set whatever threshold value you want*. If I'm deciding who to bet on to win a race, and some bookie tells me there's a 25% chance racer $$A$$ wins the race, I can *decide* to put him in the "not going to win the race" category, but that's because I *chose* 25% as at or below my decision threshold. I could choose my threshold to be 10% and therefore declare that racer $$A$$ will win the race. It's up to me.

A logit function is also unlike $$n$$th-degree polynomial regressors, however, in that its range is restricted from $$0$$ to $$1$$. This might clue you into the fact that polynomial regressors do not consider predictions conditional probabilities, and can therefore output any real number, which implies they don't output probabilities.

Logistic regression can also be used in multiclass classification, called multinomial logistic regression, where for $$n$$ labels you create $$n-1$$ logit models with one class as a reference.




