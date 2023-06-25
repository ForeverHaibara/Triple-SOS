import numpy as np
import sympy as sp
from sympy.core.singleton import S

from ...utils.polytools import verify_hom_cyclic, deg
from ...utils.basis_generator import generate_expr, arraylize, arraylize_sp
from ...utils.expression.cyclic import CyclicSum, CyclicProduct

a, b, c = sp.symbols('a b c')

class LinearBasis():
    is_cyc = False
    def __init__(self, expr_ = None) -> None:
        self.expr_ = expr_
        self.array_ = None
        self.array_sp_ = None

    @property
    def _expr_(self):
        return self.expr_

    @property
    def expr(self):
        if self.expr_ is not None:
            return self.expr_
        return self._expr_

    @property
    def _array_(self):
        self.array_ = arraylize(self.expr.doit().as_poly(a,b,c), cyc = self.is_cyc)
        return self.array_

    @property
    def array(self):
        if self.array_ is not None:
            return self.array_
        return self._array_

    @property
    def _array_sp_(self):
        self.array_sp_ = arraylize_sp(self.expr.doit().as_poly(a,b,c), cyc = self.is_cyc)
        return self.array_sp_

    @property
    def array_sp(self):
        if self.array_sp_ is not None:
            return self.array_sp_
        return self._array_sp_

    def __str__(self) -> str:
        return str(self.expr)

    def __repr__(self) -> str:
        return str(self.expr)


class LinearBasisCyclic(LinearBasis):
    is_cyc = True
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # @property
    # def _array_(self):
    #     # assert isinstance(self.expr, CyclicSum)
    #     self.array_ = arraylize(self.expr.args[0], cyc = True)
    #     return self.array_

    # @property
    # def _array_sp_(self):
    #     # assert isinstance(self.expr, CyclicSum)
    #     self.array_sp_ = arraylize_sp(self.expr.args[0], cyc = True)
    #     return self.array_sp_



class LinearBasisSquare(LinearBasisCyclic):
    r"""
    \sum_{cyc} (a-b)^(2i) * (b-c)^(2j) * (c-a)^(2k) * a^m * b^n * c^p * (tangent)
    """

    _cached_poly_square = {}

    def __init__(self, i, j, k, m, n, p, tangent = None, tangent_is_cyc = None) -> None:
        super().__init__()
        self.info_ = (i, j, k, m, n, p)

        if tangent is not None:
            self.tangent_ = tangent
            if tangent_is_cyc is None:
                tangent_is_cyc = verify_hom_cyclic(tangent.doit().as_poly(a,b,c))[1]
        else:
            self.tangent_ = S.One
            tangent_is_cyc = True
        self.tangent_is_cyc_ = tangent_is_cyc

    @property
    def tangent(self):
        return self.tangent_

    @property
    def _expr_(self):
        i, j, k, m, n, p = self.info_
        if i == j and j == k:
            self.expr_ = CyclicProduct((a-b)**(2*i)) * CyclicSum(a**m * b**n * c**p * self.tangent_, evaluate = False)
        else:
            self.expr_ = CyclicSum(
                (a-b)**(2*i) * (b-c)**(2*j) * (c-a)**(2*k) * a**m * b**n * c**p * self.tangent_
            )
        return self.expr_

    @classmethod
    def generate(cls, degree, tangent = None, tangent_is_cyc = None):
        """
        Generate all possible expressions with degree = degree, i.e.
        2*i + 2*j + 2*k + m + n + p + deg(tangent) = degree

        Also, to reduce cyclic expression, we have i >= k.
        """
        _cached_poly_square = cls._cached_poly_square

        rets = []

        if tangent is not None:
            tangent_poly = tangent.doit().as_poly(a,b,c)
            if tangent_is_cyc is None:
                tangent_is_cyc = verify_hom_cyclic(tangent_poly)[1]

            degree -= deg(tangent_poly)

            def _mul_poly(poly, m, n, p):
                return poly * a**m * b**n * c**p * tangent_poly
        else:
            def _mul_poly(poly, m, n, p):
                return poly * a**m * b**n * c**p
        
        if degree < 0:
            return rets

        for p1 in range(degree // 2 + 1):
            p2 = degree - p1 * 2
            # p1 = i + j + k, p2 = m + n + p

            for i, j, k in generate_expr(p1, cyc = tangent_is_cyc)[1]:
                poly_ijk = _cached_poly_square.get((i,j,k))
                if poly_ijk is None:
                    poly_ijk = ((a-b)**(2*i) * (b-c)**(2*j) * (c-a)**(2*k)).as_poly(a,b,c)
                    _cached_poly_square[(i,j,k)] = poly_ijk

                for m, n, p in generate_expr(p2, cyc = not (i == j and j == k))[1]:
                    rets.append(cls(i, j, k, m, n, p, tangent, tangent_is_cyc))
                    rets[-1].array_ = arraylize(_mul_poly(poly_ijk, m, n, p), expand_cyc = True, cyc = True)

        return rets



class CachedCommonLinearBasisSquare():
    common_tangents = [
        None,
        (a**2 - b*c)**2,
        (a**3 - b*c**2)**2,
        (a**3 - b**2*c)**2,
    ]

    _cached_common_linear_basis = {}

    @classmethod
    def generate(cls, degree):
        if degree in cls._cached_common_linear_basis:
            return cls._cached_common_linear_basis[degree]

        rets = []
        for tangent in cls.common_tangents:
            rets += LinearBasisSquare.generate(degree, tangent = tangent)

        cls._cached_common_linear_basis[degree] = rets
        return rets



class LinearBasisAMGM(LinearBasisCyclic):
    r"""
    \sum_{cyc} a^(i+1)*b^(j)*c^(k-1) + a^(i)*b^(j+1)*c^(k-1) + a^(i-1)*b^(j-1)*c^(k+2) - 3*a^(i)*b^(j)*c^(k)
    """
    _cached_basis = {}

    def __init__(self, i, j, k):
        super().__init__()
        self.info_ = (i, j, k)
    
    @property
    def _expr_(self):
        i, j, k = self.info_
        self.expr_ = CyclicSum(
            a**(i+1)*b**j*c**(k-1) + a**i*b**(j+1)*c**(k-1) + a**(i-1)*b**(j-1)*c**(k+2) - 3*a**i*b**j*c**k
        )
        return self.expr_
    
    @property
    def _array_sp_(self):
        i, j, k = self.info_
        inv_monoms = generate_expr(i + j + k, cyc = True)[0]
        self.array_sp_ = np.zeros(len(inv_monoms))
        for coeff, monom in zip((1,1,1,-3), ((i+1,j,k-1), (i,j+1,k-1), (i-1,j-1,k+2), (i,j,k))):
            for monom_perm in ((monom), (monom[1], monom[2], monom[0]), (monom[2], monom[0], monom[1])):
                t = inv_monoms.get(monom_perm)
                if t is not None:
                    self.array_sp_[t] += coeff
        return self.array_sp_

    @property
    def _array_sp_(self):
        i, j, k = self.info_
        inv_monoms = generate_expr(i + j + k, cyc = True)[0]
        self.array_sp_ = sp.zeros(len(inv_monoms), 1)
        for coeff, monom in zip((1,1,1,-3), ((i+1,j,k-1), (i,j+1,k-1), (i-1,j-1,k+2), (i,j,k))):
            for monom_perm in ((monom), (monom[1], monom[2], monom[0]), (monom[2], monom[0], monom[1])):
                t = inv_monoms.get(monom_perm)
                if t is not None:
                    self.array_sp_[t] += coeff
        return self.array_sp_

    @classmethod
    def generate(cls, degree):
        if degree in cls._cached_basis:
            return cls._cached_basis[degree]

        rets = []
        for i in range(1, degree - 1):
            for j in range(1, degree - i):
                k = degree - i - j
                rets.append(cls(i, j, k))

        cls._cached_basis[degree] = rets
        return rets


class CachedCommonLinearBasisSpecial():
    _cached_basis = {
        3: [
            CyclicSum(a*(a-b)*(a-c)),
            CyclicSum(a**2*b-a*b*c),
            CyclicSum(a**2*c-a*b*c),
        ],
        6: [
            CyclicProduct(a) * CyclicSum(a*(a-b)*(a-c)),
            CyclicProduct(a) * CyclicSum(a**2*b-a*b*c),
            CyclicProduct(a) * CyclicSum(a**2*c-a*b*c),
            CyclicSum(a**2*(a**2-b**2)*(a**2-c**2)),
            CyclicSum(a*b*(a*b-a*c)*(a*b-b*c)), 
            CyclicSum(a*(a-b)*(a-c)) ** 2,
            CyclicSum(a**4*(a-b)*(a-c))
        ]
    }

    @classmethod
    def generate(cls, degree):
        if degree in cls._cached_basis:
            if not isinstance(cls._cached_basis[degree][0], LinearBasis):
                cls._cached_basis[degree] = [LinearBasisCyclic(x) for x in cls._cached_basis[degree]]

        else:
            # register new basis
            basis = []
            if degree >= 3:
                # Schur
                basis.append(CyclicSum(a**(degree - 2)*(a-b)*(a-c)))

                if degree >= 7:
                    if degree % 2 == 1:
                        basis.append(CyclicSum(
                            a**(degree//2-2)*(a-b)*(a-c)) * CyclicSum(a**(degree//2-1)*(a-b)*(a-c))
                        )
                    else:
                        basis.append(CyclicSum(a**(degree//2-2)*(a-b)*(a-c)) ** 2)

            cls._cached_basis[degree] = [LinearBasisCyclic(x) for x in basis]

        return cls._cached_basis[degree]