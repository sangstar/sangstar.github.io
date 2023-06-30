---
layout: post
title:  "Words as vectors"
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

Naive Bayes or logistic regression can be perfectly suitable tools for things like text classification, but the way it handles words is antiquated. Modern methods eschew Bayesian approaches of modeling $$P(c \mid d)$$ in favor of leveraging more "modern" machine learning techniques like neural networks, which build the basis for even more advanced structures like LSTMs and transformers. But neural networks operate with vectors of real numbers, so how do we reconcile this when dealing with words? 

## Representing words with numbers
In sentiment analysis, one of the important things to note is that different words have different **connotations** -- happy has a positive connotation while sad has a negative one. Words that are similar in *meaning* can have different connotations. "Thrifty" and "cheap" both associate someone with some degree of frugality, but the former admits a positive evaluation while the latter a negative one. This, in the context of NLP, can be known as *affective meanings*. Osgood et al. all the way back on 1957 laid the groundwork for vector semantics when he found that affective meaning for a word followed 3 different characteristics: 

- The pleasantness of a stimulus, otherwise known as *valence*. "Joyful" is high in valence, while "indifferent" is more neutral, and "gloomy" is low in valence.
- The intensity of emotion provoked by the stimiulus, otherwise known as *arousal*. "Exhilarating" would be high in arousal, while "stimulating" moderately, and "soothing" low. 
- The degree of control exerted by the stimulus, otherwise known as *dominance*. "Authoritative" is high in dominance, while "influential" is moderately dominant, and "submissive" would be not particularly dominant. 

You can thereby represent a word by the three different dimensions here by assigning numeric scores for a word for each of the dimensions. 

$$
\begin{array}{cc} 
&
\begin{array}{ccccc} Valence & Arousal & Dominance\\
\end{array}
\\
\begin{matrix}
word 1 \\ word 2 \\ word 3 \\
\end{matrix}
&
\left[
\begin{array}{ccccc}
7 & 2 & 1.2 \\
1 & 3.5 & 3 \\
9.07 & 1.02 & 8.5 \\
\end{array}
\right]
\end{array}
$$