from encoder.bch.bchcodegenerator import BchCodeGenerator
from encoder.bch.bchcoder import BchCoder
import numpy as np
from sympy import Poly
from sympy.abc import x, alpha

bch_generator = BchCodeGenerator(n=255,
                             d=5,
                             field_poly_coeffs=[1, 0, 0, 0, 1, 1, 1, 0, 1])

P_x, G_x = bch_generator.gen()

bch_coder = BchCoder(n=255,
                     d=5,
                     r_poly=P_x,
                     g_poly=G_x)

input_arr = np.random.randint(0,2,239)
print(input_arr)
encoded = bch_coder.encode(msg_poly=Poly(input_arr[::-1], x))
print(len(encoded))
print(encoded)

print("final g(x)={}".format(G_x))
print("final p(x)={}".format(P_x))

