---
layout: post
title:  "Writing C++ Extensions for Python with LibTorch"
date:   2024-04-15 17:15
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

## Writing C++ Extensions for Python with LibTorch
The reference implementation of Python is in C, so whenever you're running Python code, you're usually running C code under-the-hood. The compiled bytecode itself that you've heard the interpreter creates is all managed and executed canonically in CPython in `compile.c` and `ceval.c` in the Python source code. This naturally lends to the question of whether you can write C code to be run with Python, and the answer is, unsurprisingly, yes. In fact, most Python libraries that are even somewhat concerned with performance usually have a C extension that *extends* Python with your auxiliary library. This is usually accomplished by compiling your extension in to one or many shared object files, which can be imported directly by Python if done correctly. It can even be written in C++, as long as it adheres to the Python C API. In fact, you should really write any Python C extensions in C++ because you're able to use `pybind11` or `nanobind` to create your bindings for you rather than having to painstakingly interface with the Python C API itself (trust me, don't try this).

C/C++ extensions being supported is a large part of why even the privileged Python programmer isn't immune from having to suffer through failed builds. You've no doubt had to google some hard-to-understand error from trying to install `torch` from source at one point, perhaps. The internet is littered with these kinds of pasted tracebacks. By having C extensions in your code, installing your Python app now involves compiling C code, and this now enables new degrees of freedom for build failures, such as:

- The CPython version your extension is built
- The C/C++ ABI version the code was compiled with being compatible with the ABI version used by your CPython version
- The operating system the code was compiled for (Mac, Linux, etc)
- The computer architecture your code was compiled for (amd64 or arm64, for instance)
