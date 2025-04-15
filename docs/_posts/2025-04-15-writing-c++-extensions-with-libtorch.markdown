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
cmake_minimum_required(VERSION 3.30)
project(src)

list(APPEND CMAKE_PREFIX_PATH "<path-to-libtorch>")

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
set(CMAKE_CXX_STANDARD 20)

add_library(
        adder
        SHARED
        src/bindings.cpp
)

target_link_libraries(adder
        PRIVATE
        "${TORCH_LIBRARIES}"
        Python3::Python
        <path-to-libtorch_python.dylib>
)

set_target_properties(adder PROPERTIES
        PREFIX ""            
        SUFFIX ".so"         
        OUTPUT_NAME "adder"
)

set(PYTHON_SITE_PACKAGES "${Python3_ROOT_DIR}/lib/python3.10/site-packages")

add_custom_command(
    TARGET adder
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:adder>" "${PYTHON_SITE_PACKAGES}/"
    COMMENT "Copying adder.so to ${PYTHON_SITE_PACKAGES}"
)
```

I'll highlight some of the key parts here that makes the shared object file for my extension build successfully.

First, I added:

```
list(APPEND CMAKE_PREFIX_PATH "<path-to-libtorch>")
```

because I wanted to guarantee that `cmake` could find my `libtorch` installation by adding it to the list of directories it looks for when I call `find_package()`.


I added these two lines:

```
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
```

To ensure that my Python and `torch` header files could be located. I also made sure to, when adding the library for my extension `adder`, to make sure it's a `SHARED` library to ensure it can be loaded dynamically at runtime, which Python needs it to be able to do. 

Finally, I have:

```
target_link_libraries(adder
        PRIVATE
        "${TORCH_LIBRARIES}"
        Python3::Python
        <path-to-libtorch_python.dylib>
)
```

To ensure that my library, which references headers from CPython and LibTorch, is able to link these libraries at runtime. I specifically had to include the `libtorch_python.dylib` file as well, which didn't seem to be included by `$TORCH_LIBRARIES`, which helps `pybind11` to manage the conversion between Python and LibTorch tensors and other PyTorch types. 

The other parts of my CMake afterwards just make sure to copy my shared object file in my `site-packages` directly which is sanitary (as long as there are no naming conficts with file and other packages) and guarantees Python can locate it when importing.

## Writing your extension code
The hard part is done. Now I can write some LibTorch code for my Python package. Here's a simplistic example.

```
// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

torch::Tensor my_add(const torch::Tensor& a, const torch::Tensor& b) {
    return a + b;
}

PYBIND11_MODULE(adder, m) {
    m.def("my_add", &my_add, "Add two tensors");
}
```

Note that the module name in `PYBIND11_MODULE`, `adder`, *must* match the name of the `.so` file you build. It might be immediately apparent how simple `pybind11` makes things. It simply asks for the module you want to add and the objects you want to add. It, along with the help of `torch/extension.h`, will handle the type interop between the two languages. I neglected to include a separate header and source file for the definition and implementation of `my_add` and just slapped it on the `bindings.cpp` file directly, although usually it's better to do this unless you have a really small extension module. 

In this example, my module is simply adding a function `my_add`. It takes two `const torch::Tensor&` `a` and `b`, and returns a `torch::Tensor` that is their sum. `torch::Tensor` is actually an alias for `at::Tensor`, the low-level LibTorch tensor object, and has an `operator+` defined already for these types, so this addition function is really easy to accomplish. 

Now, I didn't necessarily have to specify that `a` and `b` were `const torch::Tensor&`. I could've just made them `torch::Tensor&` or even `torch::Tensor`. Passing them by reference is a no-brainer considering how large tensors can be, but it's also a really good idea to promise you won't mutate them with `const`. As a general rule of thumb, you should not be modifying Python objects in C/C++ extensions directly. Unless you hold the GIL, which you don't want to do, mutating it won't be threadsafe, since you'll be fighting with the Python runtime trying to reference it while it's silently mutating. It's generally recommended that if you want to modify something that you copy it so that data integrity is maintained and everything is kept threadsafe. 

Now that the bindings are written, I can finally build my extension and invoke my module `adder`:

```
import torch
import adder

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

result = adder.my_add(a, b)

# Result: tensor([5., 7., 9.])
```

As a final question: was the GIL released for this or was this performed sequentially? The answer, in fact, is no. `pybind11` doesn't do this automatically for you. Here's how I could modify my function in `bindings.cpp` to release the GIL.

```
// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <stdlib.h>

namespace py = pybind11;

void my_add(const torch::Tensor& a, const torch::Tensor& b) {
    py::gil_scoped_release release;

    auto added = a + b;

    py::gil_scoped_acquire acquire;
    std::cout << "My tensors added equal " << a + b << std::endl;
}

PYBIND11_MODULE(adder, m) {
    m.def("my_add", &my_add, "Add two tensors");
}
```

In this example, I release the GIL to compute `a + b`, then reacquire it to print the result. I reacquired it here so that Python's `stdout` buffer doesn't race with `std::cout` when printing to `stdout`. I don't have any Python code writing to `stdout` in my example, but it's just a little way to show acquiring the GIL too. Another little detail here that may justify acquring the GIL before doing `std::cout <<` is in case you're printing something that is a direct `py::object` type. The override for `operator<<` for `py::object` is to call `py::str`, which directly calls the `__str__` method for that object in Python. Since it's running Python code, *it needs the GIL*, and if it doesn't have it it'll hang until it can.
