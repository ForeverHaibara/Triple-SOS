from functools import wraps

class mark:
    """Mark a problem with hints."""

    skip = 'skip'
    """Skip a problem due to computation or resource limits."""

    noimpl = 'noimpl'
    """Unimplemented problem due to difficult modelling or other issues."""

    nvars = 'nvars'
    """N-variate inequality."""

    recur = 'recur'
    """Recursive inequality."""

    geom = 'geom'
    """Geometric inequality."""

    quant = 'quant'
    """Inequalities involving non-trivial quantifiers."""

    def __new__(cls, *mark_args):
        """Decorate a problem with given marks."""
        def decorator(func):
            func.marks = mark_args
            # print(func.__name__, func.marks)
            return func
        return decorator




class ProblemSet:
    """
    A collection of problems from a source.

    * books:
        Inequalities from books. Highly structured. Many 3 or 4 variable inequalities
        over real or positive real numbers.

    * forums:
        Inequalities from forums. Some problems are much harder than those in books.
        Solutions to these problems are often computer-aided and are not suitable
        by hands.

    * contests:
        Inequalities from contests are often raw, and often require suitable
        preprocessing or transformations to simplify the problem. Some algebraic
        inequalities are obtained by unpleasant reformulations of the original
        problems. Below are some examples.
            * The constraint "a1,a2,a3,a4,a5" are a permutation of "1,2,3,4,5" might be
            formulated as the zero-dim ideal `a1^k+a2^k+a3^k+a4^k+a5^k = 1^k+2^k+3^k+4^k+5^k`
            for `k=1,2,3,4,5`.
            * The constraint "there exists i,j,k such that ai+aj+ak>=0" might be
            formulated as `Max_{i,j,k}(ai+aj+ak)>=0`, involving O(n^3) terms.
            * A geometric inequality involing the distances between 4 coplanar points A,B,C,D
            might introduce a Cayley-Menger determinant as the constraint.

        These problems might introduce many new variables and constraints in
        the reformulation, which make the problem harder to solve. Additionally,
        some contest problems might involve thousands of variables or constraints, which will
        cause time and space complexity issues. These problems should not be expanded
        to (dense) polynomials and should be solved by operating on the symbolic tree only.
    """
    @classmethod
    def collect(cls, include=tuple(), exclude=tuple()):
        cands = {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}
        include = (include,) if isinstance(include, str) else include
        exclude = (exclude,) if isinstance(exclude, str) else exclude
        for m in include:
            dels = []
            for k, v in cands.items():
                if m not in getattr(v, 'marks', tuple()):
                    dels.append(k)
            [cands.pop(k) for k in dels]
        for m in exclude:
            dels = []
            for k, v in cands.items():
                if m in getattr(v, 'marks', tuple()):
                    dels.append(k)
            [cands.pop(k) for k in dels]
        return cands
