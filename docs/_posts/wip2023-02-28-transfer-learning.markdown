---
layout: post
title:  "Transfer learning"
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


# The types of transfer learning
There are actually three: instance-based transfer, feature-representation transfer and parameter transfer.

In instance-based transfer, you use some of the data used to train the pre-trained model to assist in training your own model. In feature representation transfer, hidden layer activations from the pre-trained model are used to aid the input features with the learned patterns of the pre-trained layer, either replacing them entirely or concatenating themselves to the input vector. 

I will go out on a limb and say parameter transfer is the most common form that I've encountered. It involves using the weights from a pre-trained model to assist in the learning process for your model. 

This usually involves adding some fully-connected layers on top of the pre-trained model, and then either freezing the weights on the old layers or letting them change while introducing new data relevant to your use case, often with two different learning rates for the old and new layers. The learning rate for the new layers ought to be much higher as you want them to change much more drastically compared to the weights from the old layers, because the whole point of parameter transfer is the assumption that those weights are already pretty good. 

In transfer learning the pre-trained model's learned weights and embeddings effectively have done most of the work for you if your target task is relatable. Fewer datapoints are needed to train this model because the hope is if the target task is relatable, less datapoints will be "wasted" learning fundamental representations from both the source and target task, and allow the data to allow for more "advanced" insights. It's kind of like having a goal of teaching students calculus in as few lessons as possible and noting that high school students will require less lessons to accomplish this goal than kindergarteners because the high school students won't have to learn many basic math concepts the younger cohort will. 

## References

Sarle, W. S. (n.d.). Comp.ai.neural-Nets FAQ, part 2 of 7: Learning. faqs.org. Retrieved February 26, 2023, from http://www.faqs.org/faqs/ai-faq/neural-nets/part2/ 

Olah, C. (n.d.). Neural networks, manifolds, and topology. Neural Networks, Manifolds, and Topology -- colah's blog. Retrieved February 26, 2023, from https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ 