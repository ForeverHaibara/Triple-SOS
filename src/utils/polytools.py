from collections import defaultdict
from itertools import product
from math import ceil as ceiling
from typing import List, Tuple, Dict

import sympy as sp
from scipy.spatial import ConvexHull

from .basis_generator import generate_expr

def deg(poly: sp.Poly) -> int:
    """
    Return the degree of a polynomial. We shall assume the polynomial is homogeneous.

    Since the polynomial is homogeneous, we can simply return the sum of the exponents of the first monomial.
    This avoids calling `monoms` and significantly improves the speed.
    """
    # rep = poly.rep.rep
    # n = len(rep) + len(rep[0]) + len(rep[0][0]) - 3
    # # only exception is poly == 0 (n = -1)
    # return max(n, 0)

    # too slow:
    # return sum(poly.monoms()[0])
    return poly.total_degree()


def verify(y, polys, poly, tol: float = 1e-10) -> bool:
    """
    Verify whether the fraction approximation is valid
    by substracting the partial sums and checking whether the remainder is zero.

    Deprecated.
    """
    try:
        for coeff, f in zip(y, polys):
            if coeff[0] != 0:
                if coeff[1] != -1:
                    if not isinstance(coeff[0], sp.Expr):
                        poly = poly - sp.Rational(coeff[0], coeff[1]) * f
                    else:
                        v = coeff[0] / coeff[1]
                        coeff_dom = preprocessText_getdomain(str(v))
                        if coeff_dom != sp.QQ:
                            v = sp.polys.polytools.Poly(str(v)+'+a', domain=coeff_dom)\
                                 - sp.polys.polytools.Poly('a',domain=coeff_dom)
                        poly = poly - v * f
                else:
                    poly = poly - coeff[0] * f

        for coeff in poly.coeffs():
            # some coefficient is larger than tolerance, approximation failed
            if abs(coeff) > tol:
                return False
        return True
    except:
        return False


def verify_isstrict(func, root, tol=1e-9):
    """
    Verify whether a root is strict.

    Warning: Better get the function regularized beforehand.

    Deprecated.
    """
    return abs(func(root)) < tol


def verify_hom_cyclic(poly: sp.Poly, fast=True) -> Tuple[bool, bool]:
    """
    Check whether a polynomial is homogenous and 3-var-cyclic

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be checked.
    fast : bool
        Whether to use a faster algorithm. Default is True. It is currently in experimental stage. 
        The faster algorithm is based on direct operations on poly.rep.rep.

    Returns
    ----------
    is_hom : bool
        Whether the polynomial is homogenous.
    is_cyc : bool
        Whether the polynomial is 3-var-cyclic.
    """
    if fast:
        rep = poly.rep.to_list()
        len0, len1, len2 = len(rep), len(rep[0]), len(rep[0][0])
        n = len0 + len1 + len2 - 3
        if n < 1:
            return True, True
        
        t = n + 1
        coeffs = [0] * t**2

        for i in range(len(rep)):
            xi = rep[i]
            if len(xi):
                monom_i = len0 - i - 1
                v = monom_i * t + len(xi)
                rem = n - monom_i - len(xi)
                for j in range(len(xi)):
                    xj = xi[j]
                    v -= 1 # this is a pointer to `coeffs`
                    # monom_j = len(xi) - j - 1
                    rem += 1 # rem = n - monom_i - monom_j
                    if len(xj):
                        monom_k = len(xj) - 1
                        if monom_k != rem or any(xj[1:]):
                            return False, False
                        coeffs[v] = xj[0]

    else:
        # slow algorithm, abandoned
        monoms = poly.monoms()
        n = sum(monoms[0])

        if len(poly.gens) != 3:
            for monom in monoms:
                if sum(monom) != n:
                    return False, False
            return True, False

        t = n + 1
        coeffs = [0] * t**2 # for i in range(t**2)]
        for coeff, monom in zip(poly.coeffs(), monoms):
            if sum(monom) != n:
                return False, False
            coeffs[monom[0]*t + monom[1]] = coeff
        
    for i in range((n-1)//3+1, n+1):
        # 0 <= k = n-i-j <= i
        for j in range(max(0,n-2*i), min(i+1,n-i+1)):
            # a^i * b^j * c^{n-i-j}
            k = n-i-j
            u = coeffs[i*t + j]
            v = coeffs[j*t + k]
            if u == v:
                w = coeffs[k*t + i]
                if u == w:
                    continue 
            return True, False 
    return True, True


def verify_is_symmetric(poly: sp.Poly) -> bool:
    """
    Check whether a polynomial is symmetric. The polynomial is 
    assumed to be cyclic.
    """
    coeffs = {}
    for monom, coeff in poly.terms():
        coeffs[monom] = coeff
    for monom, coeff in coeffs.items():
        a, b, c = monom
        if coeffs.get((b, a, c), 0) != coeff:
            return False
    return True


def monom_of(x: sp.Poly, m: Tuple[int, ...]) -> sp.Expr:
    return x.coeff_monomial(m)


def convex_hull_poly(poly: sp.Poly) -> Tuple[Dict[Tuple[int, ...], bool], List[Tuple[int, ...]]]:
    """
    Compute the convex hull of a polynomial.

    Parameters
    ----------
    poly : sympy.Poly
        The polynomial to be processed.

    Returns
    ----------
    convex_hull : Dict[Tuple[int, ...], bool]
        A dictionary of the convex hull. The key is a tuple of three integers, representing a monomial.
        The value is a boolean, indicating whether the monomial is in the convex hull.
    vertices : List[Tuple[int, ...]]
        The vertices of the convex hull.
    """
    nvars = len(poly.gens)
    monoms = poly.monoms()
    n = sum(monoms[0])    # degree

    if nvars == 3 and verify_hom_cyclic(poly)[1]:
        # 3-var and cyclic
        monoms = monoms[::-1]

        skirt = monoms[0][0]  # when (abc)^n | poly, then skirt = n
        # print(skirt)

        convex_hull = [(i, j, n-i-j) for i , j in product(range(n+1), repeat = 2) if i+j <= n]
        convex_hull = dict(zip(convex_hull, [True for i in range((n+1)*(n+2)//2)]))

        vertices = [(monoms[0][1]-skirt, monoms[0][0]-skirt)]
        if vertices[0][0] != 0:
            line = skirt
            for monom in monoms:
                if monom[0] > line:
                    line = monom[0]
                    vertices.append((monom[1]-skirt, monom[0]-skirt)) # (x,y) in Cartesian coordinate
                if monom[1] == skirt:
                    break

            # remove the vertices above the line
            k , b = - vertices[-1][1] / vertices[0][0] , vertices[-1][1] # y = kx + b
            vertices = [vertices[0]]\
                    + [vertex for vertex in vertices[1:-1] if vertex[1] < k*vertex[0] + b]\
                    + [vertices[-1]]

            if len(vertices) > 2:
                hull = ConvexHull(vertices, incremental = False)
                vertices = [vertices[i] for i in hull.vertices] # counterclockwise

            # place the y-axis in the front
            i = vertices.index((0, b))
            vertices = vertices[i:] + vertices[:i]
            # print(vertices)

            # check each point whether in the convex hull
            i = -1
            for x in range(vertices[-1][0] + 1):
                if x > vertices[i+1][0]:
                    i = i + 1
                    k = (vertices[i][1] - vertices[i+1][1]) / (vertices[i][0] - vertices[i+1][0])
                    b = vertices[i][1] - k * vertices[i][0]
                # (x, y) = a^(skirt+y) b^(skirt+x) c^(n-2skirt-x-y)  (y < kx + b) is outside the convex hull
                t = skirt + x
                for y in range(skirt, skirt+ceiling(k * x + b - 1e-10)):
                    convex_hull[(y, t, n-t-y)] = False
                    convex_hull[(t, n-t-y, y)] = False
                    convex_hull[(n-t-y, y, t)] = False

        # outside the skirt is outside the convex hull
        for k in range(skirt):
            for i in range(k, (n-k)//2 + 1):
                t = n - i - k
                convex_hull[(i, t, k)] = False
                convex_hull[(i, k, t)] = False
                convex_hull[(k, i, t)] = False
                convex_hull[(k, t, i)] = False
                convex_hull[(t, i, k)] = False
                convex_hull[(t, k, i)] = False
        
        vertices = [(skirt+y, skirt+x, n-2*skirt-x-y) for x, y in vertices]
        vertices += [(j,k,i) for i,j,k in vertices] + [(k,i,j) for i,j,k in vertices]
        vertices = set(vertices)
        return convex_hull, vertices
    else:
        raise NotImplementedError("Not implemented for non 3-var&cyclic polynomials.")

        is_hom = poly.is_homogeneous
        f = (lambda x: x) if not is_hom else (lambda x: x[:-1])
        invf = (lambda x: x) if not is_hom else (lambda x: x + (n - sum(x),))

        vertices = [f(monom) for monom in monoms]
        hull = ConvexHull(vertices, incremental = False)
        vertices = [invf(vertices[i]) for i in hull.vertices]
        convex_hull = defaultdict(lambda: False)

        # compute all integers in the convex hull
        ...

        # for vertex in vertices:
        #     convex_hull[f(vertex)] = True
        return convex_hull, vertices