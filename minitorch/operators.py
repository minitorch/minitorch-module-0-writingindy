"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    """
    Multiplies two numbers

    Args:
        x: Number of float type
        y: Number of float type
    
    Returns:
        Value of x multipled by y
    """
    return x * y

def id(x: float) -> float:
    """
    Returns the input unchanged

    Args:
        x: Input of float type
    
    Returns:
        Input x
    """
    return x

def add(a: float, b: float) -> float:
    """
    Adds two numbers

    Args:
        a: Number of float type
        b: Number of float type
    
    Returns:
        Value of a added with b
    """
    return a + b

def neg(x: float) -> float:
    """
    Negates a number

    Args:
        x: Number of float type

    Returns:
        Value of x multipled by -1
    """
    return -1 * x

def lt(x: float, y: float) -> bool:
    """
    Checks if one number is less than another

    Args:
        x: Number of float type
        y: Number of float type

    Returns:
        Truth value of whether x is less than y
    """
    return x < y

def eq(x: float, y: float) -> bool:
    """
    Checks if two numbers are equal

    Args:
        x: Number of float type
        y: Number of float type

    Returns:
        Truth value of whether x is equal to y
    """
    return x == y

def max(x: float, y: float) -> float:
    """
    Returns the larger of two numbers

    Args:
        x: Number of float type
        y: Number of float type

    Returns:
        Either the number x or y, depending on which is larger
    """
    if lt(x, y):
        return y
    else:
        return x
    
def is_close(x: float, y: float) -> bool:
    """
    Checks if the two numbers are within 1e-5 of each other

    Args:
        x: Number of float type
        y: Number of float type
    
    Returns:
        Truth value of whether x and y are within 1e-5 of each other
    """
    if abs(x - y) < 1e-2:
        return True
    else:
        return False
    

def inv(x: float) -> float:
    """
    Computes the reciprocal of a number

    Args:
        x: Number of float type

    Returns:
        Reciprocal of number x
    """
    return 1 / x

def inv_back(x: float, y:float) -> float:
    """
    Computes the derivative of reciprocal of variable times a second arg
    
    Args:
        x: Number of float type, variable you take derivative of
        y: Number of float type

    Returns:
        Value of derivative of reciprocal of variable x times number y
    """
    return -y/x**2

def log_back(x: float, y: float) -> float:
    """
    Computes the derivative of log of variable times a second arg

    Args:
        x: Number of float type, variable you take derivative of
        y: Number of float type
    
    Returns:
        Value of derivative of log of variable x times number y
    """
    return y/x

def relu(x: float) -> float:
    """
    Applies the ReLU activation function

    Args:
        x: Number of float type, variable to apply ReLU activation function to

    Returns:
        Value of ReLU activation function applied to variable x
    """
    if x > 0:
        return x
    else:
        return 0

def relu_back(x: float, y: float) -> float:
    """
    Computes the derivative of ReLU of variable times a second arg

    Args:
        x: Number of float type, variable to apply ReLU activation function to
        y: Number of float type

    Returns:
        Value of derivative of ReLu of variable x times number y
    """
    if x > 0:
        return y
    else:
        return 0
    
def sigmoid(x: float) -> float:
    """
    Computes the sigmoid function of variable
    
    Args:
        x: Number of float type, variable to apply sigmoid function to

    Returns:
        Value of sigmoid function applied to variable x
    """
    if x < 0:
        return math.exp(x)/(1 + math.exp(x))
    else:
        return 1/(1 + math.exp(-x))

def log(x: float) -> float:
    """
    Computes the natural logarithm of number

    Args:
        x: Number of float type
    
    Returns:
        Value of natural logarithm of number x
    """
    return math.log(x)

def exp(x: float) -> float:
    """
    Computes the exponential of number

    Args:
        x: Number of float type
    
    Returns:
        Value of exponential of number x
    """
    return math.exp(x)

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:

    def map_fn(input: Iterable[float]) -> Iterable[float]:
        res=[]

        for el in input:
            res.append(fn(el))
        return res
    
    return map_fn

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:

    def zipWith_fn(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        res = []
        
        done=False

        iter1 = iter(ls1)
        iter2 = iter(ls2)

        while done is False:
            el1 = next(iter1, None)
            el2 = next(iter2, None)

            if el1 is None or el2 is None:
                done = True
            else:
                res.append(fn(el1, el2))

        return res
        
    return zipWith_fn


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:

    def reduce_fn(ls: Iterable[float]) -> float:
        res = start

        for el in ls:
            res = fn(el, res)

        return res
    
    return reduce_fn


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0)(ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1)(ls)