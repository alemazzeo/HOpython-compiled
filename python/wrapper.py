# -*- coding: utf-8 -*-

import ctypes as C
import numpy as np
math = C.CDLL('../src/libmymath.so')

int_p = C.POINTER(C.c_int)
float_p = C.POINTER(C.c_float)

math.add_float.restype = C.c_float
math.add_float.argtypes = [C.c_float, C.c_float]

math.add_int.restype = C.c_int
math.add_int.argtypes = [C.c_int, C.c_int]

math.add_float_ref.argtypes = [float_p, float_p, float_p]

math.add_int_ref.argtypes = [int_p, int_p, int_p]

math.dot_product.restype = C.c_float


def add_int_array(a, b):
    n = len(a)
    a_c = np.asarray(a, dtype=C.c_int)
    b_c = np.asarray(b, dtype=C.c_int)
    res = np.zeros(n, dtype=C.c_int)

    math.add_int_array(a_c.ctypes.data_as(int_p),
                       b_c.ctypes.data_as(int_p),
                       res.ctypes.data_as(int_p),
                       C.c_int(n))
    return res


def dot_product(a, b):
    n = len(a)
    a_c = np.asarray(a, dtype=C.c_float)
    b_c = np.asarray(b, dtype=C.c_float)

    res = math.dot_product(a_c.ctypes.data_as(float_p),
                           b_c.ctypes.data_as(float_p),
                           C.c_int(n))

    return res