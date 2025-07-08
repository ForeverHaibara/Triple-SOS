from random import shuffle, seed, randint, choices
from sympy.abc import a,b,c,d,x,y,z,w
from sympy.combinatorics import Permutation, PermutationGroup, DihedralGroup, CyclicGroup
from sympy.core import Function, Symbol
from sympy.simplify import signsimp
from sympy.testing.pytest import slow
from ..cyclic import CyclicSum, SymmetricSum, rewrite_symmetry

def _get_random_perm_group(degree, perm_count):
    perms = []
    for i in range(perm_count):
        perm = list(range(degree))
        shuffle(perm)
        perms.append(Permutation(perm))
    perm_group = PermutationGroup(*perms)
    return perm_group

def _compare_replacement(val, replacement):
    val1 = val.doit().xreplace(replacement)
    val2 = val.xreplace(replacement).doit()
    return signsimp(val1 - val2).expand() == 0

def test_cyclic_sum_doit():
    assert CyclicSum(x+2, (x,), PermutationGroup(Permutation([0]))) == x + 2

    assert CyclicSum(a**3*b**2*c, (x,y,z,w), DihedralGroup(4)) == 8*a**3*b**2*c

    val1 = CyclicSum(a**2*b, (c,a,b), PermutationGroup(Permutation([1,0,2]))).doit()
    val2 = a**2*b + b*c**2
    assert val1.doit() == val2

    val1 = CyclicSum(d**3*a**2*c, (d,c,a,b), DihedralGroup(4))
    val2 = a**3*b*d**2 + a**3*c*d**2 + a**2*b*d**3 + a**2*c*d**3\
          + a*b**3*c**2 + a*b**2*c**3 + b**3*c**2*d + b**2*c**3*d
    assert val1.doit() == val2


@slow
def test_cyclic_sum_random_xreplace():
    seed(2024)
    for n in range(6):
        symbols = tuple(Symbol('a%d'%i) for i in range(n))
        func = Function("F")(*symbols)
        for perm_count in range(1, min(n, 6)):
            for rep in range(30):
                perm_group = _get_random_perm_group(n, perm_count)
                # perm_group2 = _get_random_perm_group(n, perm_count)
                # val = CyclicSum(func * CyclicSum(func, symbols, perm_group2), symbols, perm_group)
                val = CyclicSum(func, symbols, perm_group)

                subset = set(choices(symbols, k = randint(1, min(n, 4))))
                subset = dict((s, 7 + i) for i, s in enumerate(subset))
                assert _compare_replacement(val, subset), '%s(%s).xreplace(%s)'%(val.func, val.args, subset)

def test_cyclic_sum_xreplace_preserve_structure():
    F = Function('F')
    G = Function('G')
    val = CyclicSum(d**3*a**2*c, (a,b,c,d), DihedralGroup(4))

    replacements = [
        {a:x, b:c, c:z, d:b},
        {a:G(a), b:G(b), c:G(c), d:G(d)},
        {a:a**2*(b+d), b:b**2*(a+c), c:c**2*(b+d), d:d**2*(a+c)},
        {a:G(b+d+b*d), b:G(a+c+a*c), c:G(b+d+b*d), d:G(a+c+a*c)}
    ]
    for replacement in replacements:
        assert len(val.xreplace(replacement).find(CyclicSum))
        assert _compare_replacement(val, replacement)

    p2 = PermutationGroup(Permutation(3,4,5)(0,1,2),Permutation(4,3),Permutation(0,1))
    val = CyclicSum(F(a,b,c,d,x,y,z), (x,y,z,a,b,c), p2)
    assert val.xreplace({F(b): b}) == val

def test_cyclic_sum_xreplace_partial():
    F = Function('F')
    val = CyclicSum(a**3*b**2*c, (a,b,c,d), 
        PermutationGroup(Permutation([1,0,2,3]),Permutation([0,1,3,2])))
    replacements = [
        {c: d + 5},
        {a: a+b, b: a+b},
        {a: F(a*b), b: F(a*b), c:2*c+2*d+a+b, d:2*c+2*d+a+b},
        {a: c**2+d**2, b: c**2+d**2},
        {a: F(a), b: 101}
    ]
    for replacement in replacements:
        assert len(val.xreplace(replacement).find(CyclicSum))
        assert _compare_replacement(val, replacement)

    replacements = [
        {c: F(a,b,c,d)},
        {a: b+2*c+d}
    ]
    for replacement in replacements:
        assert _compare_replacement(val, replacement)

def test_cyclic_sum_xreplace_nonsymbol():
    F = Function('F')
    val = CyclicSum(F(a), (a,b,c), CyclicGroup(3))
    assert _compare_replacement(val, {F(a): y})
    
    val = CyclicSum(F(a), (a,b,c,d), DihedralGroup(4))
    assert _compare_replacement(val, {F(a): x})
    assert _compare_replacement(val, {F(b): y})
    assert _compare_replacement(val, {F(b): d})

    val = CyclicSum(a, (a,b,c), CyclicGroup(3))
    assert val.xreplace({val: x, z: y}) == x

    val = SymmetricSum(a*(b+c+2), (a,b,c))
    val2 = SymmetricSum(F(a)*(F(b)+F(c)+2), (a,b,c))
    assert val.xreplace({a:F(a), b:F(b), c:F(c), w:F(a+b+c)}) == val2

    val = CyclicSum(F(a,b,c,x,y,z), (x,y,z,a,b,c),
            PermutationGroup(Permutation(3,4,5)(0,1,2),Permutation(4,3,size=6),Permutation(0,1,size=6)))
    assert val.xreplace({F(a,b,c): x}) == val


def test_rewrite_symmetry():
    val = SymmetricSum(a*(b**2+c), (a,b,c))

    for arr in ([1,0,2], [0,1,2]):
        p1 = PermutationGroup(Permutation(arr))
        val2 = rewrite_symmetry(val, (a,b,c), p1)
        assert (val - val2).doit().expand() == 0

    p0 = PermutationGroup(
        Permutation([1,2,0,4,5,3]),
        Permutation([3,4,5,0,1,2]),
        Permutation([1,0,2,4,3,5])
    )
    val = CyclicSum(a*b**2*c**3*x**4*y**5*z**6, (a,b,c,x,y,z))
    for p1 in p0.args:
        val2 = rewrite_symmetry(val, (a,b,c,x,y,z), PermutationGroup(p1))
        assert (val - val2).doit().expand() == 0
