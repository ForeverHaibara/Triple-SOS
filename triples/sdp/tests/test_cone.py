from sympy import MutableDenseMatrix as Matrix

from ..cone import ConeSDPBuilder


def test_cone_agm():
    # |x|^p + 2|y|^p <= 3|z|^p where z = 1
    for p in range(2, 11):
        builder = ConeSDPBuilder(3)
        builder.add_pnorm_cone(
            Matrix([[1,0,0],[0,1,0]]),
            Matrix([0,0,1]),
            p=p,
            weight=Matrix([1, 2])/3
        )
        sdp = builder.build()

        sol = sdp.solve_obj([4, -5] + [0]*(sdp.dof - 2),
                            [sdp.gens[2] - 1])

        if p == 1:
            sol0 = (-3, 0, 1)
        else:
            x0 = (3/(1 + 2*(5/8)**(p/(p-1))))**(1/p)
            y0 = x0*(5/8)**(1/(p-1))
            sol0 = (-x0, y0, 1)

        assert max((sol[:3,:] - Matrix(list(sol0))).applyfunc(abs)) < 1e-4,\
            f"error when p = {p}, expected solution {sol0}, but got {tuple(sol[:3,:])}"
