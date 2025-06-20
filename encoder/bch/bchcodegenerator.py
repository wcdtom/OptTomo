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
            :param d: distance of codes, d = 2*t + 1 < 2^m - 1
            r_poly: primary polynomial or field polynomial
            bg_poly: generation polynomial
            m: q^m-1 = n, e.g., for BCH(255,239) q=2, n=255, m=8
                m is also the highest order of r_poly
            k: information bit length, e.g., for BCH(255,239), k=239
                k = 2^m - 1 - mt
            t: error correction capability, e.g., for BCH(255,239), t=2
        '''
        self.n = n
        self.d = d
        self.q = 2
        self.m = order(self.q, self.n)
        self.field_poly_coeffs = field_poly_coeffs
        log.info("BchCodeGenerator(n={},q={},m={},d={}) initiated"
                  .format(self.n, self.q, self.m, self.d))

    def gen(self):
        irr_poly = Poly(self.field_poly_coeffs, alpha)
        g_poly = None
        for i in range(1, self.d):
            if g_poly is None:
                g_poly = minimal_poly(i, self.n, self.q, irr_poly)
            else:
                g_poly = lcm(g_poly, minimal_poly(i, self.n, self.q, irr_poly))
        g_poly = g_poly.trunc(self.q)
        log.info("g(x)={}".format(g_poly))
        return irr_poly, g_poly
