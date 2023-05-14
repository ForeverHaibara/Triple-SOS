from .peeling import _make_coeffs_helper
from .sparse  import _sos_struct_sparse
from .quartic import _sos_struct_quartic
from .quintic import _sos_struct_quintic
from .sextic  import _sos_struct_sextic
from .septic  import _sos_struct_septic
from .octic   import _sos_struct_octic


def _sos_handle_branch(result):
    if result is not None:
        return result
    return [], None, None


def SOS_Special(poly, degree, ext = False):
    """
    SOS for special structures.

    Params
    -------
    poly: sp.polys.polytools.Poly
        The target polynomial.

    degree: int
        Degree of the polynomial.

    Returns
    -------
    multipliers: list of str
        The multipliers.
        
    y: list of tuples
        Rational coefficients of each component.

    names:

    """
    coeff, coeffs = _make_coeffs_helper(poly, degree)
    
    if len(coeffs) == 1 and poly.monoms()[0] == (0,0,0): # zero polynomial
        return [], [(0,1)], [f'a^{degree}+b^{degree}+c^{degree}']

    if len(coeffs) <= 6: # commonly Muirhead or AM-GM or trivial ones
        multipliers, y, names = _sos_handle_branch(
                                    _sos_struct_sparse(poly, degree, coeff, recurrsion = SOS_Special, coeffs = coeffs)
                                )
        if y is not None:
            return multipliers, y, names
        else:
            return None

    multipliers, y, names = None, None, None

    if degree == 4:
        multipliers, y, names = _sos_handle_branch(
                                    _sos_struct_quartic(poly, degree, coeff, recurrsion = SOS_Special)
                                )

    elif degree == 5:
        multipliers, y, names = _sos_handle_branch(
                                    _sos_struct_quintic(poly, degree, coeff, recurrsion = SOS_Special)
                                )

    elif degree == 6:
        multipliers, y, names = _sos_handle_branch(
                                    _sos_struct_sextic(poly, degree, coeff, recurrsion = SOS_Special)
                                )

    elif degree == 7:
        multipliers, y, names = _sos_handle_branch(
                                    _sos_struct_septic(poly, degree, coeff, recurrsion = SOS_Special)
                                )

    elif degree == 8:
        multipliers, y, names = _sos_handle_branch(
                                    _sos_struct_octic(poly, degree, coeff, recurrsion = SOS_Special)
                                )


    if (y is None) or (names is None) or len(y) == 0:
        return None

    y = [x.as_numer_denom() if not isinstance(x, tuple) else x for x in y]
    return multipliers, y, names