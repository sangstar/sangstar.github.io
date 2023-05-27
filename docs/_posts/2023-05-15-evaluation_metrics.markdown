---
layout: post
title:  "Evaluation metrics in NLP"
date:   2023-05-15 10:15
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

I've written about some important things when it comes to evaluation in previous articles, but I wanted to dedicate one solely to it as it's probably the most important area of knowledge to be proficient in when using ML to solve problems in the real world. Without clear ways to judge the performance of a model and its performance against others designed to tackle the same problem, your model will not (or at least should not) make it out of a dev environment. I will be covering a way to do both. 


## Precision and Recall
In my article on [the misconceptions of the imbalanced dataset](https://sangstar.github.io/ml/2023/04/22/imbalanced-datasets.html) I make it clear that accuracy is typically an undesirable metric for classification. For the latter, the two big metrics of note are *precision* and *recall*. Precision measures the percentage of items that the model detected (labeled as positive) that are actually positive (true positive and false positives ), 

$$ \text{Precision} = \frac{t_p}{t_p + f_p}$$

(where $$t_p$$ and $$f_p$$ corresponds to true and false positive respectively) while recall measures the percentage of correctly identified inputs:

$$ \text{Recall}  = \frac{t_p}{t_p + f_n}$$

where $$f_n$$ represents false negatives (the denominator represents all positive cases -- true positives and false negatives both being all positives). Consider the wonderful "nothing is about our pie" classifier from Jurafsky and Martin's *Speech and Language Processing* (my favorite textbook on NLP) which is a model that predicts every input -- trained on a million tweets unrelated to (0) or specifically discussing his/her love or hatred about pie made by the fictional *Delicious Pie Company* (1) -- as "not about our pie". In this example, only 100 of the million samples actually discuss the pie. Accuracy is defined as the percentage of true classifications overall:

$$\text{Accuracy} = \frac{t_p + t_n}{t_p + f_p + t_n + f_n}$$

In this case, a "nothing is about our pie" classifier would have $$999,900$$ true negatives and $$100$$ false negatives, boasting an accuracy of $$99.99\%$$. Yet it would have $$0$$ true positives (never classifying anything as about *Delicious Pie Company* pie), and therefore would have a recall of $$0$$! Not so potent now. The precision would similarly also be $$0$$. The true positives being alone in the numerator, unable to be buffered by true negatives like in accuracy is what makes the two metrics so popular -- we typically in business cases are more concerned with true positives (ability to detect some important outcome like whether there were tweets about pie) than overall accuracy.

## Confusion matrices
A really useful tool for evaluating classification performance is from developing a *confusion matrix*. A confusion matrix is a matrix with the following structure:

$$
\begin{bmatrix}
t_p & f_p \\
f_p & t_n \\
\end{bmatrix}
$$

which can be extended quite easily to multiple classes by just adding more columns and rows.

$$
\begin{array}{cc} 
&
\begin{array}{ccccc} A & B & C\\
\end{array}
\\
\begin{matrix}
\hat A \\ \hat B \\ \hat C \\
\end{matrix}
&
\left[
\begin{array}{ccccc}
15 & 2 & 0 \\
1 & 35 & 3 \\
22 & 0 & 25 \\
\end{array}
\right]
\end{array}
$$

In this example, the row labels corresond to the predictions by the models by its associated row elements, and the column labels are the "true" or "gold" labels. Rows off the diagonal, such as that marked by $$1$$, would be a false positive label for $$B$$, where the correct label is $$A$$. "False positive" and "false negative" kind of break down with more classes, so it's more just saying that it's either on the diagonal and a true positive, or a misclassification of a gold label. 

You may have noticed that a classifier with minimal misclassification tends to diagonality, and it makes for a cool way to visually check how your classifier is doing in terms of precision and recall *per class*. For example, the model is poor at classifying $$C$$, but strong at classifying $$B$$ and $$A$$.


## F-measure

You'll commonly want to combine the metrics instead of working with two. Probably the most notable combination is the *F-measure*. The *harmonic mean* of a set of numbers $$\{v_i\}_{i=1}^n$$ is:

$$H = \frac{n}{\frac{1}{v_1} + \frac{1}{v_2} + ... + \frac{1}{v_n}}$$

and since we're combining two metrics by averaging in some way, keeping in mind that they're ratios, this is a good place to start. The harmonic mean for precision and recall can be written as:

$$F = \frac{1}{\alpha \frac{1}{P} + \gamma \frac{1}{R}}$$

Where the constants are used to weight the metrics by just having them appear more times in the sum. $$\gamma$$ is traditionally set to $$\gamma = 1 - \alpha$$ which renders

$$F = \frac{1}{\alpha \frac{1}{P} + (1-\alpha) \frac{1}{R}}$$

and simplifying further, multiplying all terms by $$PR$$:

$$ F = \frac{PR}{\alpha R + (1-\alpha) P} $$

multiplying all terms by $$\frac{1-\alpha}{\alpha}$$..

$$ \frac{\frac{1-\alpha}{\alpha}PR}{(1-\alpha)R + \frac{(1-\alpha)^2}{\alpha}P}$$

$$\frac{\frac{1-\alpha}{\alpha}PR}{(1-\alpha)(R + \frac{(1-\alpha)}{\alpha}P)}$$

$$\frac{\frac{1}{\alpha}PR}{R + \frac{(1-\alpha)}{\alpha}P}$$

Then, making the substitution $$\beta^2 = \frac{1-\alpha}{\alpha}$$:

$$\frac{\frac{1}{\alpha}PR}{R + \beta^2 P}$$

and noting that $$\beta^2 + 1 = \frac{1}{\alpha}$$:


$$F_\beta = \frac{(\beta^2 + 1)PR}{\beta^2 P + R}$$

$$\beta^2$$ is now a weighting factor that you can use to prefer precision or recall. $$\beta^2 < 1$$ favors precision, while $$\beta^2 > 1$$ favors recall. It's important to understand why. As $$\beta^2 \to 0$$, we have $$\beta^2P + R \to R$$, and $$\beta^2 + 1 \to 1$$, so we are left with something converging to 

$$\lim_{\beta^2 \to 0} F_\beta = \frac{PR}{R} = P$$

conversely if $$\beta^2 \ge 1$$ and as $$\beta^2 \to \infty$$ we have $$\beta^2 P + R \to \beta^2 P$$ and $$(\beta^2 + 1) \to \beta^2$$ so we are left with 

$$\lim_{\beta^2 \to \infty} F_\beta = \frac{PR}{P} = R$$



## Statistical Significance
When trying to work out if model $$A$$ is superior to model $$B$$, comparing them on one test set is bad practice and would be unacceptable evidence in most bodies of scientific literature. You will need to enter the domain of statistical hypothesis testing. 

Suppose we want to compare the model performances of model $$A$$, a recurrent neural network (RNN), to model $$B$$, a naive Bayes sentiment classifier on a test set $$T$$. 

Suppose you take the $$F_1$$ scores of both models on the test set  (let's say $$M(A, T)$$ for performance by $$A$$ on test set $$T$$ and $$M(B, T)$$ for performance by $$B$$ on test set $$T$$) and define the performance difference as:

$$
\delta(T) = M(A,T) - M(B,T)
$$

And more specifically for $$F_1$$ scores:

$$
\delta(T) = F_1(A,T) - F_1(B,T)
$$

This performance difference is known as an *effect size*. Suppose $$\delta(T) = 0.2$$. According to statistical hypothesis testing, this *does not* allow us to state that $$A$$ is a superior model to $$B$$. That's because it's entirely possible that $$A$$ was "accidentally" better than $$B$$ on test set $$T$$, and that $$B$$ is actually no worse than $$A$$ or even better.  

In the realm of statistical inference, to make a claim that some agent $$A$$ outperforms $$B$$, we make two hypotheses: that $$A$$ is either as good or worse than $$B$$, or that $$A$$ is better than $$B$$. 

The first hypothesis is the one we assume is true, known as the null hypothesis. 

$$H_0 : \delta(T) \le 0$$

The goal to proving statistical significance in $$A$$'s superiority to $$B$$ is to find the empirical probability that we'd find our value of $$\delta(T)$$ or of one even greater if the null hypothesis is true. Basically we want to find the probability that we would see $$\delta(T)$$ or higher if $$A$$ is actually not better than $$B$$ with regard to some test statistic (in our case $$F_1$$). That is to say that we want to find for some arbitrary test set $$t$$:

$$p = P(\delta(t) \ge \delta(T) \ | \ H_0 \ \text{is true})$$

This probability is called a *p-value*. You might've seen the p-value before in studies and stuff, where it's usually set to something like $$0.05$$, which means there is a $$95\%$$ chance the null hypothesis is not true: that whatever $$B$$ is (in studies talking about medical interventions this is often called a placebo!) cannot be considered on par with or better than $$A$$ at some test statistic, like the difference in $$F_1$$ scores or the difference in amyloid deposition in the brain for an Alzheimer's drug. It's an *incredibly powerful* statistic. The result of $$A$$ being better than $$B$$ is *statistically significant* if the probability we defined is below a threshold we decided on, such as $$p \le 0.05$$. 

In NLP, we approach computing this probability $$p$$ generally using non-parametric tests, (as parametric tests don't generally work with our data because data in NLP is rarely normal) such as a bootstrap test, which I'm going to be discussing here. 

## Paired Bootstrap Test
Bootstrapping is a word in statistics used to describe random sampling *with replacement*, which basically means once a datapoint is sampled, that datapoint has an equal likelihood of being sampled as it had before. It's not out of the available pool of datapoints to be sampled. It's like if you were to sample a ball out of a bag of red, green and blue balls, but once you took one out of the bag, before you sampled it again you'd put that ball back. It's actually also quite important in decision trees, and is not an unreasonable thing to do as long as you are able to assume that the bootstrapped sample is representative, which is a fairly reliable assumption to make with a large enough sample size, the data collection process not introducing any bias or systematic errors that would be propogated, and the data being independent and identically distributed (ie not dealing with weird/complex data structures with spatial, temporal or hierarchical dependencies). 


## References
Jurafsky, D., & Martin, J. H. (2019). Naive Bayes, Text Classification and Sentiment. In Speech and Language Processing (3rd ed., Chapter 4). Prentice Hall.
