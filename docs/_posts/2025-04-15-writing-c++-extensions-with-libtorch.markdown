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

C/C++ extensions being supported is a large part of why even the privileged Python programmer isn't immune from having to suffer through failed build tracebacks. You've no doubt had to Google some hard-to-understand error from trying to install `torch` from source at one point, perhaps. The internet is littered with these kinds of pasted error messages. By having C extensions in your code, installing your Python package now involves compiling C code, and this now enables new degrees of freedom for build failures, such as having to be mindful of:

- The CPython version your extension is built with
- The C/C++ ABI version the code was compiled with being compatible with the ABI version used by your CPython version
- The operating system the code was compiled for (Mac, Linux, etc)
- The computer architecture your code was compiled for (amd64 or aarch64, for instance)

*All* of which have to be compatible for the user. It essentially inherits all of the same woes building software from a compiled language has. 

NumPy has C extensions, which is why it's so fast, but you've probably never had any issues installing it. That's because `pip` found the correct wheel for your machine and installed that, which has all of the code precompiled for your target architecture. However, if no wheel exists for your target architecture (ARM users definitely can relate), then you're left building from source. 

Compiling CUDA code is even more burdensome, because it has even more degrees of freedom for failed builds, such as:

- Driver compatibility (such as a driver being too old for compiled PTX)
- Kernels being compiled for the right GPU arch (e.g. `sm70`, `sm90`, etc)
- Any weirdness involved with dynamically loading the CUDA runtime (`libcudart.so`) or any other important dynamic libraries

That's why `torch` wheels have such scary names, which may, adhering to the PEP 3149 standard, have to specify the `torch` version, the CUDA build it's built for, the CPython version, the C++ ABI version, 
and the platform all in one.

```
torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl
```

Anyway, let's get back on track here, and talk about LibTorch.

## LibTorch
Calling C extension code in Python is very beneficial. For one, C/C++ is generally way faster than Python as is the case generally when comparing compiled languages with interpreted ones, and because Python, with its dynamic typing, automatic memory management, metadata-rich everything-is-an-object philosopy, lack of inlining paired with large overhead for calling functions, and copious support for runtime checks like reflection, in particular heavily prioritizes ease of use over performance (although this could warrant its own article in the first place).

Crucially, Python code *cannot natively run CPU-bound tasks concurrently*. The Global Interpreter Lock (GIL) forbids this, since its memory management model relies on incrementing and decrementing object references to qualify an object for garbage collection, and this isn't thread-safe. All of these things point to a fairly agreed upon idiom: when writing performant code in Python, your Python code should be wrapping over bindings from a compiled language as much as possible. Unless you're mutating objects that are directly in scope in the Python runtime, you can even manually release the GIL and return control to Python when your binding code is called. 

`torch` is no exception to this idiom, and the C/C++ API for `torch` is `LibTorch`, which the Python module `torch` actually wraps over. With it, you can actually write C extensions using PyTorch's C++ API. 

In order to extend CPython with your C/C++ code, you eventually need to interop with the Python C API. This dreadful task is managed for you by binding libraries like `pybind11`, which LibTorch has taken the liberty of creating bindings for its data structures (like `torch::Tensor`) so that `pybind11` knows how to handle the conversions going from a Pythonic `torch.Tensor` to a LibTorch `torch::Tensor`, for instance. So, how do you get started?

## Getting your bindings set up
Building your extension should be done with `cmake`. Here's an example of
a `CMakeLists.txt` for one of my projects:

```cmake
dwdawd
```



