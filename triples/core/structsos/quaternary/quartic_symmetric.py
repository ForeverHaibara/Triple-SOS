def quaternary_quartic_symmetric(coeff, real=True):
    sol = _quaternary_quartic_symmetric_sos(coeff)
    if sol is None:
        return None
    return sol

def _quaternary_quartic_symmetric_sos(coeff):
    """
    Solve symmetric quartic polynomials representable by
    sum-of-squares.

    Examples
    --------
    :: sym = "sym"

    => s(2a2-3ab)2

    => 1/5s(ab)2

    => s((a-b)2(a+b-4c-4d)2)

    => s(a2b2-a2bc)

    => s(a4-a3b-a2bc+2a2b2-abcd)

    => s(26a4+(236-244*sqrt(2))a3b+(373-198*sqrt(2))a2b2+(1444-1036*sqrt(2))a2bc+(297-202*sqrt(2))abcd)
    """
    from ..nvars import _sos_struct_nvars_quartic_symmetric_sdp
    return _sos_struct_nvars_quartic_symmetric_sdp(coeff)
