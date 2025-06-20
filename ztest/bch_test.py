from encoder.bch.bchcodegenerator import BchCodeGenerator
from encoder.bch.bchcoder import BchCoder
from encoder.bch.padding import *
import numpy as np
from sympy import Poly
from sympy.abc import x, alpha
from encoder.bch.mathutils import *

bch_generator = BchCodeGenerator(n=255,
                             d=5,
                             field_poly_coeffs=[1, 0, 0, 0, 1, 1, 1, 0, 1])

P_x, G_x = bch_generator.gen()

bch_coder = BchCoder(n=255,
                     d=5,
                     r_poly=P_x,
                     g_poly=G_x)

input_arr = np.random.randint(0,2,231)
padded_tx = padding(input_arr, 239)
# print(padded_tx)
encoded = bch_coder.encode(msg_poly=Poly(padded_tx, x))
encoded = np.array(encoded)
# print(encoded)
padded_rx = padding(encoded, 255)
# print(padded_rx)
# print(len(padded_rx))

# pow_dict = power_dict(n=255,
#                      irr=P_x,
#                      p=2)
# print(pow_dict)

decoded = bch_coder.decode(msg_poly=Poly(encoded,x))

# print(decoded)
print(np.array_equal(padded_tx, decoded))
# print("final g(x)={}".format(G_x))
# print("final p(x)={}".format(P_x))

