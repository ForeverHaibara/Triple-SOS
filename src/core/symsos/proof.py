import sympy as sp

def _prove_irreducible_univariate(coeff):
    """
    Prove an irreducible univariate polynomial is non-negative.

    TODO:
    1. Handle real case and positive case.
    2. Use complete algorithm.
    """
    if not coeff.is_Add:
        return coeff

    gens = tuple(coeff.free_symbols)
    if len(gens) != 2:
        return None
    x, y = gens
    poly = coeff.subs(y, 1).as_poly(x)

    all_coeffs = poly.all_coeffs()
    if all_coeffs[0] < 0 or all_coeffs[-1] < 0:
        return None

    if len(all_coeffs) == 3:
        # quadratic polynomial
        a, b, c = all_coeffs
        if b*b - 4*a*c > 0:
            return None

        return a*(x + b/(2*a)*y)**2 + (c - b**2/(4*a)) * y**2

    if all(_ >= 0 for _ in all_coeffs):
        return coeff

    return None

def _prove_coeff(coeff, factorized = True):
    """
    Given a coefficient, prove that it is non-negative. The coeff
    should be homogeneous with respect to x and y.
    """
    if not factorized:
        coeff = coeff.factor()

    args = []
    if coeff.is_Mul:
        args = coeff.args
    elif isinstance(coeff, (sp.Pow, sp.Add, sp.Symbol, sp.Rational)):
        args = [coeff]

    proved_args = []
    for arg in args:
        if isinstance(arg, (sp.Symbol, sp.Rational)):
            proved_args.append(arg)
        elif arg.is_Pow:
            if arg.exp % 2 == 0 or isinstance(arg.base, sp.Symbol):
                proved_args.append(arg)
            else:
                proved = _prove_irreducible_univariate(arg.base)
                if proved is None:
                    return None
                proved_args.append(proved ** arg.exp)
        elif arg.is_Add:
            proved = _prove_irreducible_univariate(arg)
            if proved is None:
                return None
            proved_args.append(proved)

    return sp.Mul(*proved_args)

def _prove_numerator(numerator):
    z = numerator.gens[0]
    args = []
    for ((deg, ), coeff) in numerator.terms():
        proved = _prove_coeff(coeff, factorized = False)
        if proved is None:
            return None
        args.append(proved * z**deg)

    return sp.Add(*args)

