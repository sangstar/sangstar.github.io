---
layout: post
title:  "Computer Architecture for ML people"
date:   2024-12-02 10:42
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

# Important (or handy) things to know about Computer Architecture for ML people

A common complaint among software engineers is the group of ML people coming in to the software
engineering field armed with a solid background in maths but without the appreciation for computer
architecture when creating software, and it can show. It may seem unimportant with how high-level
programming languages are today, like Python, but a basic understanding of computing is essential.

Here are some important things to know for ML people wanting to, at the very least, make their programs
more efficient:

1. Understand how the CPU works
If you ever ask yourself "why can't I have `torch` directly load these model tensors to my GPU? Why does it
have to go to CPU first?" this part is for you. CPUs and GPUs are entirely different processing units
and are designed to do different things. 

At a very reductionist level, CPUs are an assortment of different
electrical modules all orchestrated by a control unit. A non-exhaustive list: an arithmetic-logic unit (ALU), 
instruction registers to locate and load instructions for a program from memory, a clock essentially providing 
a pulsating activation for all of the CPU's components that, often along with a toggle, can perform a task (such as 
providing an input voltage that triggers a flip flop to store a bit, for instance), and a number of other local 
registers acting as quick-access memory (eschewing a data bus for memory transfer for things like arithmetic), all
connected by a data bus that can send outputs and input voltages to each of the components. Its goal is to execute
a sequence of stored instructions in memory (incremented by a program counter), performing an _instruction cycle._

When you want the CPU to perform a program, you could (hypothetically) pre-store the instructions in its designated 
reset vector in memory, so that it performs those things right as the CPU has power. The CPU will then 
**Fetch** an instruction from program memory. It will then **Decode** the instruction, which is split up by an op code 
that is defined by your CPU's instruction set (perhaps `01` is the op code for `LOAD_A` where `A` is some register `A`)
and binary for your n'th RAM memory address to relate to that operation. In that case, maybe `0111` can be split in to
`01` -> `LOAD_A` and `11` -> `ADDRESS 11`, which would mean "Load register A with whatever is in memory address 11".
It then finally **Execute**s the instruction. In this case, it would probably do something along the lines of sending a
logic high voltage to some `READ_ENABLE` input to allow for reading however many memory cells output for that address
(8 for RAM with an 8-bit word length, for instance -- although most RAM nowadays have gigabyte word lengths), or no
input is needed to query a read, and it just finds whatever the output is for that address. It then sends that data
through the data bus to register A and stores it there. 