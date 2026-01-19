from sympy import Poly
from sympy import MutableDenseMatrix as Matrix
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.ddm import DDM

from ....sdp import congruence
from ....utils import Coeff

def sos_struct_nvars_quadratic(poly: Poly, **kwargs):
    """
    Solve a quadratic inequality on real numbers.
    """
    coeff = Coeff(poly)
    nvars = coeff.nvars
    mat = [[0 for _ in range(nvars + 1)] for __ in range(nvars + 1)]
    for k, v in coeff.items():
        inds = []
        for i in range(nvars):
            if k[i] > 2:
                return None
            elif k[i] == 2:
                inds = (i, i)
                break
            elif k[i] == 1:
                if len(inds) == 2:
                    return None
                inds.append(i)
        if len(inds) == 1:
            inds.append(nvars)
        elif len(inds) == 0:
            inds = (nvars, nvars)
        if inds[0] == inds[1]:
            mat[inds[0]][inds[0]] = v
        else:
            mat[inds[0]][inds[1]] = v/2
            mat[inds[1]][inds[0]] = v/2

    ddm = DDM(mat, (nvars+1, nvars+1), coeff.domain)
    mat = Matrix._fromrep(DomainMatrix.from_rep(ddm))
    res = congruence(mat)
    if res is None:
        return None
    U, S = res
    gens = poly.gens
    genvec = Matrix(list(gens) + [1])
    return sum(S[i] * (U[i, :] * genvec)[0,0]**2 for i in range(nvars + 1))
