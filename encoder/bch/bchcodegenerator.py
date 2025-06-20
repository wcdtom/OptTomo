from encoder.bch.mathutils import *
from sympy.polys.galoistools import gf_irreducible, gf_irreducible_p
from sympy import lcm, ZZ
import logging

log = logging.getLogger("bchcodegenerator")


class BchCodeGenerator:

    def __init__(self, n, d, field_poly_coeffs):
        '''

            :param n: total bit length = information bits + BCH conding bits
                          e.g., for BCH(255,239), n=255=239+16
            :param b:
            :param d: distance of codes, d >= 2*t + 1 < 2^m - 1
            r_poly: primary polynomial or field polynomial
            bg_poly: generation polynomial
            m: q^m-1 = n, e.g., for BCH(255,239) q=2, n=255, m=8
                m is also the highest order of r_poly
            k: information bit length, e.g., for BCH(255,239), k=239
                k >= 2^m - 1 - mt
            t: error correction capability, e.g., for BCH(255,239), t=2
        '''
        self.n = n
        # self.b = b
        self.d = d
        self.q = 2
        self.m = order(self.q, self.n)
        self.field_poly_coeffs = field_poly_coeffs
        # log.info("BchCodeGenerator(n={},q={},m={},b={},d={}) initiated"
        #          .format(self.n, self.q, self.m, self.b, self.d))

    def gen(self):
        irr_poly = Poly(self.field_poly_coeffs, alpha)
        # irr_poly = Poly(alpha ** self.m + alpha + 1, alpha).set_domain(GF(self.q))
        # print("initial P(x)={}".format(irr_poly))
        # print("P(x) Coeffs:")
        # print(irr_poly.all_coeffs())
        # if gf_irreducible_p([int(c) for c in irr_poly.all_coeffs()], self.q, ZZ):
        #    quotient_size = len(power_dict(self.n, irr_poly, self.q))
        #    print(power_dict(self.n, irr_poly, self.q))
        # else:
        #     quotient_size = 0
        # log.info("irr(q_size: {}): {}".format(quotient_size, irr_poly))
        # while quotient_size < self.n:
        #     irr_poly = Poly([int(c.numerator) for c in gf_irreducible(self.m, self.q, ZZ)], alpha)
        #     print(gf_irreducible(self.m, self.q, ZZ))
        #     print("Iterated P(x)={}".format(irr_poly))
        #     quotient_size = len(power_dict(self.n, irr_poly, self.q))
        #     print(power_dict(self.n, irr_poly, self.q))
        #     log.info("irr(q_size: {}): {}".format(quotient_size, irr_poly))
        g_poly = None
        # for i in range(self.b, self.b + self.d - 1):
        for i in range(1, self.d - 1):
            if g_poly is None:
                g_poly = minimal_poly(i, self.n, self.q, irr_poly)
            else:
                g_poly = lcm(g_poly, minimal_poly(i, self.n, self.q, irr_poly))
            print("Iterated G(x)={}".format(g_poly))
        g_poly = g_poly.trunc(self.q)
        log.info("g(x)={}".format(g_poly))
        return irr_poly, g_poly
