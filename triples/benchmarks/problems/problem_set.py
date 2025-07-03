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