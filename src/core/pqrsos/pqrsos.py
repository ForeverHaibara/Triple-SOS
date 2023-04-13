###########################################################################
# Compute SOS using pqr method
#
###########################################################################
from ...utils.text_process import deg


def SOS_PQR(poly):
    """
    Compute SOS using pqr method, support symmetric polys with degree <= 8
    or cyclic polys with degree <= 5
    """
    n = deg(poly)
    # if symmetric

    #
    if n > 8:
        return

    if n == 5:
        0

    else:
        0