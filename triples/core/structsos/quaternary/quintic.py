

def quaternary_quintic_symmetric(coeff):
    if coeff((5,0,0,0)) != 0:
        # not implemented
        return
    return _quaternary_quintic_symmetric_hexagon(coeff)


def _quaternary_quintic_symmetric_hexagon(coeff):
    c41, c32, c311, c221 = [coeff(_) for _ in [(4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0)]]

    if c41 < 0 or c41 + c32 < 0:
        return

