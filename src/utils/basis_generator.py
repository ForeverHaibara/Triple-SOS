from typing import Optional, Dict, List, Tuple

import numpy as np
import sympy as sp
from sympy.polys.polytools import Poly

# from root_guess import verify_isstrict
from .text_process import next_permute, cycle_expansion
from .polytools import deg



class CachedBasis():
    def __init__(self):
        self.dict_monoms = {}
        self.inv_monoms = {}
        self.dict_monoms_cyc = {}
        self.inv_monoms_cyc = {}

        self.names = {}
        self.polys = {}
        self.basis = {}
    
    def _inquire_monoms(self, n: int, cyc: bool = True):
        if cyc:
            dict_monoms, inv_monoms = self.dict_monoms_cyc, self.inv_monoms_cyc
        else:
            dict_monoms, inv_monoms = self.dict_monoms, self.inv_monoms

        if not (n in dict_monoms.keys()):
            dict_monoms[n] , inv_monoms[n] = self._generate_expr(n, cyc = cyc)
                
        dict_monom = dict_monoms[n]
        inv_monom  = inv_monoms[n]
        return dict_monom, inv_monom

    def _generate_expr(self, n: int, cyc: bool = True):
        """
        For cyclic, homogenous, 3-variable polynomials, only part of monoms are needed:

        a^i * b^j * c^k where (i>=j and i>k) or (i==j==k)

        inv_monoms records all such (i,j,k)
        dict_monom records the index of each (i,j,k) in inv_monoms
        """
        dict_monom = {}
        inv_monom = []
        if cyc:
            for i in range((n-1)//3+1, n+1):
                # 0 <= k = n - i - j < i
                for j in range(max(0,n-2*i+1), min(i+1,n-i+1)):
                    inv_monom.append((i,j,n-i-j))
                    dict_monom[inv_monom[-1]] = len(inv_monom) - 1
            
            if n % 3 == 0:
                # i = j = k = n//3
                inv_monom.append((n//3,n//3,n//3))
                dict_monom[inv_monom[-1]] = len(inv_monom) - 1
        else:
            for i in range(n, -1, -1):
                # 0 <= k = n - i - j < i
                for j in range(n-i, -1, -1):
                    inv_monom.append((i,j,n-i-j))
                    dict_monom[inv_monom[-1]] = len(inv_monom) - 1

        return dict_monom, inv_monom



_cached_basis = CachedBasis()


def _get_dict_monom_and_inv_monom(poly, dict_monom: Optional[Dict] = None, inv_monom: Optional[List] = None, cyc: bool = True):
    if not isinstance(poly, int):
        poly = deg(poly)

    if dict_monom is None or inv_monom is None:
        dict_monom, inv_monom = _cached_basis._inquire_monoms(poly, cyc = cyc)
    return dict_monom, inv_monom


def _arraylize(poly, coeffs, dict_monom, inv_monom, cyc: bool = True):
    if cyc:
        check = lambda i, j, k: i >= j and (i > k or (i == k and i == j))
    else:
        check = lambda i, j, k: True
    for coeff, monom in zip(poly.coeffs(), poly.monoms()):
        i , j , k = monom 
        if check(i, j, k):
            coeffs[dict_monom[(i,j,k)]] = coeff
    return coeffs 

def _arraylize_expand_cyc(poly, coeffs, dict_monom, inv_monom):
    n = sum(poly.monoms()[0])
    for coeff, monom in zip(poly.coeffs(), poly.monoms()):
        i , j , k = monom 
        if i == j or j == k or k == i:
            max_, min_ = max((i,j,k)), min((i,j,k))
            i, j, k = max_, n - max_ - min_, min_
            if i == j and j == k:
                coeff *= 3
        else:
            if j > i and j > k:
                i, j, k = j, k, i
            elif k > i and k > j:
                i, j, k = k, i, j
        coeffs[dict_monom[(i,j,k)]] += coeff
    # coeffs /= 3
    return coeffs


def arraylize(poly, expand_cyc = False, dict_monom: Optional[Dict] = None, inv_monom: Optional[List] = None, cyc: bool = True):
    """
    Turn a cyclic sympy polynomial into arraylike representation.
    """
    dict_monom, inv_monom = _get_dict_monom_and_inv_monom(poly, dict_monom, inv_monom, cyc = cyc)

    coeffs = np.zeros(len(inv_monom))
    if not expand_cyc:
        return _arraylize(poly, coeffs, dict_monom, inv_monom, cyc = cyc)
    else:
        return _arraylize_expand_cyc(poly, coeffs, dict_monom, inv_monom)


def arraylize_sp(poly, expand_cyc = False, dict_monom: Optional[Dict] = None, inv_monom: Optional[List] = None, cyc: bool = True):
    """
    Turn a cyclic sympy polynomial into sympy-arraylike representation.
    """
    dict_monom, inv_monom = _get_dict_monom_and_inv_monom(poly, dict_monom, inv_monom, cyc = cyc)

    coeffs = sp.zeros(len(inv_monom), 1)
    if not expand_cyc:
        return _arraylize(poly, coeffs, dict_monom, inv_monom, cyc = cyc)
    else:
        return _arraylize_expand_cyc(poly, coeffs, dict_monom, inv_monom)


def invarraylize(poly, dict_monom: Optional[Dict] = None, inv_monom: Optional[List] = None, cyc: bool = True):
    """
    Turn an array back to cyclic sympy polynomial.
    """
    n = round((np.prod(poly.shape) * 6) ** .5 - 1.5)
    dict_monom, inv_monom = _get_dict_monom_and_inv_monom(n, dict_monom, inv_monom, cyc = cyc)

    a, b, c, u, v, w = sp.symbols('a b c u v w')
    val = sp.S(0)
    if cyc:
        for monom, coeff in enumerate(poly):
            if coeff == 0:
                continue
            i , j , k = inv_monom[monom]
            if i != j or i != k or j != k:
                val += coeff * u**i * v**j * w**k
            else:
                val += coeff * u**i * v**j * w**k / 3

        poly = val.subs({u:a,v:b,w:c}) + val.subs({u:b,v:c,w:a}) + val.subs({u:c,v:a,w:b})
    else:
        for monom, coeff in enumerate(poly):
            if coeff == 0:
                continue
            i , j , k = inv_monom[monom]
            val += coeff * a**i * b**j * c**k
        poly = val

    poly = poly.as_poly(a,b,c)
    return poly


def generate_expr(*args, **kwargs):
    return _cached_basis._inquire_monoms(*args, **kwargs)
