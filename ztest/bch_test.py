from encoder.bch.bchcodegenerator import BchCodeGenerator

bch_coder = BchCodeGenerator(n=255,
                             d=5,
                             field_poly_coeffs=[1, 0, 0, 0, 1, 1, 1, 0, 1])

P_x, G_x = bch_coder.gen()

print("final g(x)={}".format(G_x))
print("final p(x)={}".format(P_x))