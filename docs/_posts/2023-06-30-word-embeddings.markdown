---
layout: post
title:  "Words as vectors"
date:   2023-06-30 10:42
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
\begin{array}{ccccc} \text{Valence} & \text{Arousal} & \text{Dominance}\\
\end{array}
\\
\begin{matrix}
\text{word 1} \\ \text{word 2} \\ \text{word 3} \\
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

Further studies in the 50s by Joos (1950), Harris (1954), and Firth (1957) posited that words can be defined by its *distribution* in language use: that words that had similar neighboring words were probably similar words. For instance: for "She [rarely/often] speaks in class" rarely or often both appear in the same placement in this sentence and other sentences like this, which conveys a semantic connection due to having a similar *distribution*. It's important to stress that words that express opposite meanings *are similar*: despite having opposite meanings, rarely and frequently appear in many of the same contexts because they both refer to a frequency. 


These concepts on assocating words using dimensions and considering words as similar if they have similar distributions led to the emergence of the concept of **vector semantics**, where we represent a word as a point in multidimensional vector space that is derived from the distributions of its word neighbors, where the vectors are known as *embeddings* (this is sometimes pedantically only given to dense vectors like those formed in word2vec, but I think it's fair to say it's used for all vector representations of words including sparse vector spaces like co-occurence spaces which I'm about to touch on), which is the subject of this article. Vector representations of words were a brilliant idea not just for leveraging it in modern ML techniques, but codifying words with vectors frees us up to do math on them in really cool ways. If I want to now see how similar two words are, I can just take the cosine similarity of them like any other two vectors, and apply norms to work out stuff like the "importance" of a word in its space (as it'll be highly influential when combined with other words whether that's through attention, summation, averaging etc). The dimension of embedding spaces is $$V$$, known as the *vocabulary*: the number of unique words appearing in your collection of documents, called a corpus. For clarity, in NLP a *document* is a collection of words, be it a sentence, paragraph, passage, etc. Embedding spaces usually take on the form of expressing vectors based on word counts (term-term, term-document, tf-idf etc), or less interpretable, learned dense vectors (word2vec).

## Co-occurence Matrices

This method for word embeddings involve representing words based on their co-occurence with other documents, known as term-document matrices, or with other words, known as term-term matrices.

# Term-document matrices

To start with, let's talk about a term-document matrix. This matrix is constructed by having words as rows in the vocabulary $$\{w_i\}_{i=1}^V$$ and columns as documents in the corpus $$\{d_i\}_{i=1}^N$$

An example is below:


$$
\begin{array}{cc} 
&
\begin{array}{ccccc} d_1 & d_2 & d_3\\
\end{array}
\\
\begin{matrix}
w_1 \\ w_2 \\ w_3 \\
\end{matrix}
&
\left[
\begin{array}{ccccc}
1 & 0 & 25 \\
36 & 12 & 15 \\
0 & 0 & 1 \\
\end{array}
\right]
\end{array}
$$

Notice that $$0$$'s have appeared in my example. This will be a very commonly occurring characteristic to co-occurence matrices, especially with a larger corpus (which forms a larger vocab). This causes these methods to suffer from sparsity. While my matrix above isn't particularly sparse as term-document is not nearly as aggregious as term-term, sparsity is super unfavorable, as it implies a huge matrix full of mostly $$0$$'s, which feels really bad because you have a sense that the matrix's size is needlessly large, like packaging a cell phone in a 8m by 8m cardboard box. If I wanted to calculate the similarity between two word vectors of a sparse matrix, the vast majority of my operations would be $$0 \times 0$$ and $$0+0$$. Taking into account that each $$0$$ is using memory on your system, and that you get full penalty from the curse of dimensionality, this starts feeling like a huge drag, and it's partially why dense vector representations like word2vec are far more favorable. Anyway, I'll stop burying the lead. 


# Term-term matrices

Term-term matrices are predictably terms in the corpus as rows and columns. 

$$
\begin{array}{cc} 
&
\begin{array}{ccccc} w_1 & w_2 & w_3\\
\end{array}
\\
\begin{matrix}
w_1 \\ w_2 \\ w_3 \\
\end{matrix}
&
\left[
\begin{array}{ccccc}
1 & 0 & 25 \\
36 & 12 & 15 \\
0 & 0 & 1 \\
\end{array}
\right]
\end{array}
$$

It should hopefully make sense that since each word $$w_i$$ appears as a row and as a column, its term-term co-occurence is simply the amount of times it appears in the corpus, as it always appears with itself. 

Weighing by frequency is generally a good idea. While words appearing frequently together is meaningful, words like "the", "it" or "they" will have these enormously inflated embeddings, and it tends to be the case that words with the most magnitude in these vector spaces tend to be unimportant. A balance needs to be struck here, and this is where *tf-idf* comes into play.

# tf-idf
tf-idf weighting is a common approach for term-document matrices (hence the "d" in "tf-idf") to solving this problem. PPMI is common for solving this problem for term-term, but I'm going to skip this one for brevity. 

The "tf" stands for term frequency like we've dealt with before, however it's often squashed by a $$\log$$ since a logarithmic behavior is generally considered more sensible -- we want the word appearing more times to signify that the word is more relevant to the meaning of the document, but not necessarily have it be the case that the word appearing 100 times means that it's 100 times more likely to be relevant. A sense of diminishing returns is generally accepted, hence the log. We also tack on a $$+1$$ term to it so we avoid the log blowing up if there's a count of $$0$$.

$$tf = \log{(\text{count}(t,d) + 1)}$$

Cool, so now we have term frequency weighting, which we acknowledge is a good concept until we have words like "the" that will have bloated importance for appearing in so many documents. We now need to punish this behavior using **inverse document frequency**, idf. Simply, it's the inverse of a word's document frequency.

A word's document frequency ought to be how frequently it appears in a document, so should appropriately be 

$$ df_w = \frac{\text{Number of documents $$w$$ occurs in}{\text{Number of documents}}$$