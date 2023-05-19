---
layout: post
title:  "Fine-tuning an LLM and its pitfalls"
date:   2023-05-19 10:15
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

## Low-Rank Adapation of Large Language Models (LoRA)
In linear algebra, there is a concept called *low-rank approximation*, where a cost function (canonically the Frobenius norm) measures the difference between a given matrix an a matrix that approximates it of a lower rank. The task to minimize over $$\hat D$$ the following optimization problem:

$$\lVert D - \hat D \rVert _F$$

subject to $$\text{rank}(\hat D) \le r$$ where $$r$$ is some desired rank. This has an analytic solution if you represent $$D$$ using its singular value decomposition. I'm not going to go into SVD in detail for this article, but there are great resources online to read more about it. 

Representing $$D$$ into its SVD we have

$$D = U \sum V^T \in \mathbf{R}^{m \times n}, \ m \le n$$