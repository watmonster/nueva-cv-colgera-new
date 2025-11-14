#!/usr/bin/env python

import numpy as np

from example_module import a_function_from_another_module


def a_local_function():
    print("Coming to you now from a local function")
    print("I can run numpy commands!")
    print(np.array([1, 2, 3]) * 2)


def main_function():
    print("Hello, World!")
    a_local_function()
    a_function_from_another_module()


if __name__ == "__main__":
    main_function()
