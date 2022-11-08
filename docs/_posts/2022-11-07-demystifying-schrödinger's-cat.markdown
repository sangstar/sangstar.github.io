---
layout: post
title:  "Demystifying Schrödinger's Cat"
date:   2022-11-07 20:30:51 -0500
categories: physics
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


## It's not just a cat in a box!
Ever since studying quantum mechanics, I've noticed that Schrödinger's cat is mentioned often in random youtube videos I watch or people I talk to. Whenever I hear it mentioned, I almost never hear it described accurately or applied correctly in analogy. From what I can remember, this vaguely summarizes how I've heard it incorrectly spoken to me..

> The cat is in a box and you can't see him. Because you can't see him, you can never know for sure he's alive or dead. So you can never tell if something really exists unless you observe it.

Maybe it's not actually a big misconception and I've just got an unrepresentative sample of point of views. Regardless, it's made me want to clear up what actually this thought experiment is all about, and how it ultimately tries to explain the concept of *superposition states*.

## The thought experiment
Schrödinger's cat involves a cat in a box built in such a way where any outside observer has no way of seeing inside it. Then, the thought experiment involves some way of killing the cat, typically by way of a Geiger counter shattering some poisonous jar that kills the cat if the counter detects a single atomic decay. The details don't really matter, though, as long as the chances for the cat to be alive and dead are equally likely. This allows us to represent the cat's fate with a *quantum state* that can be written below.

$$\psi = \frac{1}{\sqrt{2}} |0 \rangle + \frac{1}{\sqrt{2}} |1 \rangle$$

This is known as bra-ket or Dirac notation, and is commonly used to denote what's called a *state vector* which is a basis vector in its associated Hilbert space, and more plainly something that is independent or unique from the other state vectors, such that you can't describe it using other state vectors (a concept known as linear independence). In this instance, I've denoted the 0 to refer to the state where the cat is dead and 1 to refer to where the cat is alive. Alive and dead are mutually exclusive, and cannot be used to describe eachother, so are appropriate choices for state vectors. 

The equation in one line describes what the entire thought experiment tries to describe: since the fate of the cat isn't deterministic due to the fact that we don't know when the cat will die, it cannot be described as alive or dead. A dead cat would be

$$\psi = 1 |0 \rangle$$

for instance. This represents a particle in a quantum state, where its position is "smeared" over a continuum of values until an observation is taken place, which involves multiplying the quantum state by its complex conjugate (which is kind of like squaring it) in order to render its associated positional probability distribution and rolling the dice. In other words, this cat is in a *superposition state* of being alive and dead because its state is unknowable until observation.

In terms of the cat, the likelihood of its death at any given moment doesn't chance the fact that it will never be purely alive or dead until observation, unless its death or survival is guaranteed, because we could always arrive at the first equation, just with different scalars. 

To return to the quote, it's not that aggregious. My only gripe is that the cat *could not be anything but alive* unless it had some form of lethal agent that could act at any given instant in time. Just because you can't see it doesn't mean it's not necessarily alive. The key is the fact that it's state of "alive" could change permanently at any moment.