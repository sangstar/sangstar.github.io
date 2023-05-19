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

subject to $$\text{rank}(\hat D) \le r$$ where $$r$$ is some desired rank. This has an analytic solution if you represent $$D$$ using its singular value decomposition, a generalization of the eigendecomposition of square normal matrices with an orthonormal eigenbasis to any rectangular matrix. I'm not going to go into SVD in detail for this article, but there are great resources online to read more about it. 

Representing $$D$$ into its SVD (and taking the transpose as we are not working in complex numbers) we have

$$D = U \Sigma V^T \in \mathbf{R}^{m \ \times \ n}, \ m \le n$$

Suppose we decide to partition our decomposed matrices into block matrices like so:

$$U = [U_1, U_2], \ \Sigma = diag(\Sigma_1, \Sigma_2), \ V = [V_1, V_2]$$

Where $$U_1$$ is $$m \ \times \ r$$, $$\Sigma_1$$ is $$r \ \times \ r$$ and $$V_1$$ is $$r \ \times \ n$$. These means that those associated submatrices have dominant singular values since $$\Sigma$$ typically is typically put in descending order. This restructuring captures the "important information" of matrix $$D$$ in a submatrix. If this sounds confusing, remind yourself that singular values are a rectangular analogue to eigenvalues for square matrices, and that the values are sorted by magnitude, so that $$\Sigma_1$$ holds the most heavily-weighted linear transformations of $$V$$ corresponding to $$V_1$$. 