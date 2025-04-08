class SDPError(Exception):
    """Base class for exceptions in this module."""

class SDPProblemError(SDPError):
    """Raised when the SDP is infeasible or unbounded."""

class SDPInfeasibleError(SDPProblemError):
    """Raised when the SDP is infeasible."""

class SDPUnboundedError(SDPProblemError):
    """Raised when the objective is unbounded."""

class NoInstanceMeta(type):
    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Cannot instantiate {cls.__name__!r} class")

class SDPStatusClass(metaclass=NoInstanceMeta):
    name       = 'status'
    optimal    = False
    error      = False
    inf_or_unb = False
    infeasible = False
    unbounded  = False
    error_cls  = None

class StatusOptimal(SDPStatusClass):
    name       = 'optimal'
    optimal    = True

class StatusInfOrUnb(SDPStatusClass):
    name       = 'infeasible_or_unbounded'
    inf_or_unb = True
    error_cls  = SDPProblemError

class StatusInfeasible(SDPStatusClass):
    name       = 'infeasible'
    inf_or_unb = True
    infeasible = True
    error_cls  = SDPInfeasibleError

class StatusUnbounded(SDPStatusClass):
    name       = 'unbounded'
    inf_or_unb = True
    unbounded  = True
    error_cls  = SDPUnboundedError

class StatusError(SDPStatusClass):
    name       = 'error'
    error      = True
    error_cls  = SDPError

class SDPStatus(metaclass=NoInstanceMeta):
    """Namespace for SDP statuses."""
    OPTIMAL = StatusOptimal
    INFEASIBLE_OR_UNBOUNDED = StatusInfOrUnb
    INFEASIBLE = StatusInfeasible
    UNBOUNDED = StatusUnbounded
    ERROR = StatusError


class SolverConfigs:
    verbose = 0
    tol_gap_abs = 1e-8
    tol_gap_rel = 1e-8
    tol_fsb_abs = 1e-8
    tol_fsb_rel = 1e-8
    solver_options = dict()
    _KEYS = ('verbose', 'tol_gap_abs', 'tol_gap_rel', 'tol_fsb_abs', 'tol_fsb_rel', 'solver_options')
    def __init__(self, **kwargs):
        for key in self._KEYS:
            setattr(self, key, kwargs.pop(key, getattr(self, key)))

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments for SolverConfigs: {list(kwargs.keys())}")

    def keys(self):
        return list(self._KEYS)

    def values(self):
        return [getattr(self, key) for key in self._KEYS]

    def items(self):
        return [(key, getattr(self, key)) for key in self._KEYS]

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"SolverConfigs({', '.join(f'{key}={getattr(self, key)!r}' for key in self._KEYS)})"

    def __str__(self):
        return f"SolverConfigs({', '.join(f'{key}={getattr(self, key)}' for key in self._KEYS)})"