from copy import deepcopy
from typing import Optional, Dict, List, Tuple

import numpy as np
import sympy as sp
from sympy.polys.polytools import Poly

# from root_guess import verify_isstrict
from .text_process import PreprocessText, cycle_expansion, next_permute, deg



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

    def _inquire_basis(self, n: int, tangents = ['a2-bc','a3-bc2','a3-b2c'], strict_roots = []):
        if not (n in self.names.keys()):
            dict_monom, inv_monom = self._inquire_monoms(n)
            self.names[n], self.polys[n], self.basis[n] = self._generate_basis(n, 
                tangents = tangents, strict_roots = strict_roots, 
                dict_monom = dict_monom, inv_monom = inv_monom
            )

        names, polys, basis = deepcopy(self.names[n]), deepcopy(self.polys[n]), self.basis[n].copy()
        return names, polys, basis


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


    def _generate_basis(self,
            n: int, 
            tangents: List = None,
            strict_roots: List = None,
            dict_monom: Optional[Dict] = None, 
            inv_monom: Optional[List] = None,
            aux = None
        ):
        """
        Return
        -------
        names: list of str, e.g.
            ['(a+b-c)^2']
        
        polys: list of sympy polynomials, e.g. 
            [Poly('(a+b-c)^2+(b+c-a)^2+(c+a-b)^2')]

        basis: ndarray
            Vstack of ndarray representation of each polynomials.
        """
        if dict_monom is None or inv_monom is None:
            dict_monom, inv_monom = self._inquire_monoms(n)

        names = []
        polys = []
        basis = []

        nontrivial_roots = False
        if strict_roots is not None and len(strict_roots) > 0:
            for (a, b) in strict_roots:
                if min((abs(a-1), abs(b-1), abs(a-b), abs(a), abs(b))) > 5e-3:
                    nontrivial_roots = True
                    break

        if not nontrivial_roots:
            for m in range(0,n//2+1):
                # ((a-b)^i (b-c)^j (c-a)^k)^2    where 2 <= 2(i+j+k) = 2m <= n
                # and    i+j+k = m

                # m2 = n - 2m,    standing for the remaining orders excluding the squares
                m2 = n - 2*m
                for prefix in base_square(m):
                    for suffix in base_trivial(m2):
                        names.append(prefix + ' * ' + suffix)

            for i in range(1,n-1):
                for j in range(1, n-i):
                    k = n - i - j
                    name = f'a^{i+1}*b^{j}*c^{k-1} + a^{i}*b^{j+1}*c^{k-1} + a^{i-1}*b^{j-1}*c^{k+2} - 3*a^{i}*b^{j}*c^{k}'
                    names.append(name)
                
            if aux is None:
                aux = []
            if n == 3:
                aux = ['a^2*b-a*b*c','a*b^2-a*b*c']
            elif n == 5:
                aux = []#['a*(a-b)^2*(a+b-c)^2','b*(a-b)^2*(a+b-c)^2','c*(a-b)^2*(a+b-c)^2']
            elif n == 6:
                aux = ['a*b*c*a*(a-b)*(a-c)','a^2*(a^2-b^2)*(a^2-c^2)','a*b*c*(a^2*b-a*b*c)',
                        'a*b*c*(a*b^2-a*b*c)','a*b*(a*b-b*c)*(a*b-c*a)']
            names = names + aux
            names.append(f'a^{n-2}*(a-b)*(a-c)')

            polys = [Poly(cycle_expansion(name)) for name in names]
            basis = np.array([arraylize(poly,dict_monom,inv_monom) for poly in polys])


        names, polys, basis = self._append_basis(n, tangents, dict_monom, inv_monom, names, polys, basis)

        return names, polys, basis


    def _append_basis(self, 
            n: int, 
            tangents: Optional[List[str]] = None,
            dict_monom: Optional[Dict] = None, 
            inv_monom: Optional[List] = None, 
            names: Optional[List] = None, 
            polys: Optional[List] = None, 
            basis: Optional[np.ndarray] = None
        ):
        """Add basis generated from tangents."""
        if names is None or polys is None or basis is None:
            names, polys, basis = self._inquire_basis(n)
        if not tangents:
            return names, polys, basis
        if dict_monom is None or inv_monom is None:
            dict_monom, inv_monom = self._inquire_monoms(n)

        old_names = len(names)
        for i in range(len(tangents)):
            tangents[i] = PreprocessText(str(tangents[i]), cyc=False, retText=True)

        for mixed_tangent in base_tangent(n, tangents):
            names.append(mixed_tangent)
        
        if old_names == len(names):
            return names, polys, basis

        for name in names[old_names:]:
            polys.append(Poly(cycle_expansion(name)))
            
        new_basis = np.array([arraylize(poly, dict_monom, inv_monom) for poly in polys[old_names:]])
        if type(basis) == list and len(basis) == 0:
            basis = new_basis
        else:
            basis = np.vstack((basis, new_basis))

        return names, polys, basis



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


def base_square(m: int) -> str:
    """
    (a-b)^2i * (b-c)^2j * (c-a)^2k

    where 2i+2j+2k = 2m  and  k <= i
    """
    for i in range(0, m+1):
        # 0 <= k = m-i-j <= i
        for j in range(0, m-i+1):
            prefix = f'(a-b)^{2*i} * (b-c)^{2*j} * (c-a)^{2*(m-i-j)}'
            yield prefix
            

def base_trivial(m: int) -> str:
    """
    a^i * b^j * c^k 
    where i+j+k = m  and  k <= i
    """
    for i in range((m-1)//3+1, m+1):
        # 0 <= k = m2-i2-j2 <= i2
        for j in range(max(0,m-2*i), min(i+1,m-i+1)):
            name = f'a^{i} * b^{j} * c^{m-i-j}'
            yield name
            
def base_tangent(m: int, tangents: list) -> str:
    """
    (a-b)^2i * (b-c)^2j * (c-a)^2k * a^u * b^v * c^w * f(a,b,c)^2t
    where its degree is m
    """
    for tangent in tangents:
        try:
            n = deg(Poly(tangent))
        except:
            continue
        for _ in range(3):
            for i in range(1, m//(2*n)+1):
                for j in range(0, (m-2*n*i)//2+1):
                    for prefix in base_square(j):
                        for suffix in base_trivial(m-2*(n*i+j)):
                            name = prefix + ' * ' + suffix + ' * (' + tangent + f')^{2*i}'
                            yield name
            tangent = next_permute(tangent)



def generate_expr(*args, **kwargs):
    return _cached_basis._inquire_monoms(*args, **kwargs)

def generate_basis(*args, **kwargs):
    return _cached_basis._inquire_basis(*args, **kwargs)

def append_basis(*args, **kwargs):
    return _cached_basis._append_basis(*args, **kwargs)

def reduce_basis(
        n: int, 
        strict_roots: List[Tuple], 
        names: Optional[List] = None, 
        polys: Optional[List] = None, 
        basis: Optional[np.ndarray] = None, 
        tol: float = 1e-3
    ):
    """Delete the basis that are not zero in strict roots."""
    if names is None or polys is None or basis is None:
        names, polys, basis = _cached_basis._inquire_basis(n)

    if strict_roots is None or len(strict_roots) == 0 or type(basis) == list:
        return names, polys, basis

    dict_monom, inv_monom = _get_dict_monom_and_inv_monom(n)

    m = basis.shape[0]
    mask = [1] * m

    for (a,b) in strict_roots:
        vals = []
        powera = [a**i for i in range(n+1)]
        powerb = [b**i for i in range(n+1)]
        for (i,j,k) in inv_monom:
            if i != k:
                vals.append(powera[i]*powerb[j]+powera[j]*powerb[k]+powera[k]*powerb[i])
            else: # i==j==k
                vals.append(powera[i]*powerb[j])
        vals = np.array(vals)

        vals = basis @ vals.T 
        for i in range(m):
            if abs(vals[i]) > tol: # nonzero
                mask[i] = 0
    
    mask = [i for i in range(m) if mask[i]]
    names = [names[i] for i in mask]
    polys = [polys[i] for i in mask]
    basis = basis[mask]

    return names, polys, basis