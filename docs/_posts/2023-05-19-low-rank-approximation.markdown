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
In linear algebra, there is a concept called *low-rank approximation*, where a cost function (canonically the Frobenius norm) measures the difference between a given matrix an a matrix that approximates it of a lower rank. The task is to minimize over $$\hat D$$ the following optimization problem:

$$\lVert D - \hat D \rVert _F$$

subject to $$\text{rank}(\hat D) \le r$$ where $$r$$ is some desired rank. This has an analytic solution if you represent $$D$$ using its singular value decomposition, a generalization of the eigendecomposition of square normal matrices with an orthonormal eigenbasis to any rectangular matrix. I'm not going to go into SVD in detail for this article, but there are great resources online to read more about it. 

Representing $$D$$ into its SVD (and taking the transpose as we are not working in complex numbers) we have

$$D = U \Sigma V^T \in \mathbf{R}^{m \ \times \ n}, \ m \le n$$

Suppose we decide to partition our decomposed matrices into block matrices like so:

$$U = [U_1, U_2], \ \Sigma = diag(\Sigma_1, \Sigma_2), \ V = [V_1, V_2]$$

Where $$U_1$$ is $$m \ \times \ r$$, $$\Sigma_1$$ is $$r \ \times \ r$$ and $$V_1$$ is $$r \ \times \ n$$. These means that those associated submatrices have dominant singular values since $$\Sigma$$ typically is typically put in descending order. This restructuring captures the "important information" of matrix $$D$$ in a submatrix. If this sounds confusing, remind yourself that singular values are a rectangular analogue to eigenvalues for square matrices, and that the values are sorted by magnitude, so that $$\Sigma_1$$ holds the most heavily-weighted linear transformations of $$V$$ corresponding to $$V_1$$. 

If we thereby define our matrix approximation as:

$$\hat D^* = U_1 \Sigma_1 V_1^T$$

Then by the Eckart-Young-Mirsky theorem we've achieved our goal: 

$$\lVert D - \hat D^* \rVert _F = \text{min}_{\text{rank}(\hat D) \le r} \  \lVert D - \hat D \rVert _F$$

Which allows us to approximate $$D$$ as $$\hat D$$ if we're happy with how much it's minimized (recalling that $$\lim_{x \to 0} A - B = x$$ converges to $$A = B$$ and this is what is being stated through the norm of this matrix getting smaller and smaller).

The point of all of this is that this can be applied to fine-tuning a model for a downstream task. When we fine-tune a model, we essentially for each initial (initial signified as $$i$$) weights matrix $$j$$ modify its pre-trained weights (to be clear, the weights matrix of an arbitrary initial $$i$$ pre-trained weights matrix $$j$$ which I denote as $$W_{i, j}$$) by some adjustment $$\Delta W_j$$ through training. This is traditionally nothing more than the same rank as the original weights matrix, which amounts to some pretty serious overhead with compute and memory. But, if we represented $$\Delta W_{j}$$ instead as $$B_jA_j$$, then $$BA$$ can have the required dimensions, say $$d \ \times \ k$$, while $$B$$ can have dimensions $$d \ \times \ r$$ and $$A$$ can have dimensions $$r \ \times \ k$$, where $$r$$ can be arbitrarily small. Thus, $$W_{i,j}$$ can be freezed, and $$B_j$$ and $$A_j$$ can be interpreted as additional weights matrices to learn that are rank-decompositions of $$W_{i,j}$$ such that we have the rule for updating:

$$W_{f,j} = W_{i,j} + B_j A_j$$

This is an increasingly popular way to approach the question of how to fine-tune in transfer learning. It is markedly more computationally efficient than fine-tuning with no freezing, usually blows layer-freezing few-shot learning out of the water, and has been shown to obtain results comparable or even superior to full finetuning with only a fraction of the data cost. 
