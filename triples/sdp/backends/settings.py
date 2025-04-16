from typing import Optional

import numpy as np

class SDPResult:
    """
    Class to store the result and status of the solution of an SDP.

    Attributes
    -----------
        y: Optional[np.ndarray]
            The solution or the last iterate of the SDP problem. Must be an 1D numpy array
            if not None. The solution might be returned even if the problem is infeasible
            or unbounded, so it is important to check the status `success` or `optimal`.
        success: bool
            True if the solution is optimal and accurate up to given tolerance.

        optimal: bool
            True if the solution is optimal. If `inaccurate` is True, the solution
            is optimal in the sense of reduced accuracy. To check `optimal` and not `inaccurate`,
            use the property `success`.
        infeasible: bool
            True if the problem is infeasible. If False, the problem is feasible
            or the feasibility is not determined.
        unbounded: bool
            True if the objective is feasible but unbounded. If False, the objective
            is bounded or the boundedness is not determined.
        inf_or_unb: bool
            True if the problem is infeasible or unbounded. This is equivalent to
            dual-infeasible. If False, the problem is either feasible and bounded
            or the feasibility and boundedness are not determined.
        inaccurate: bool
            A decorator flag to indicate whether the conclusions are given in reduced
            accuracy or the solver terminates early due to iteration / time limit.
        error: bool
            True if the solver encounters an unexpected internal error, e.g. wrong
            input types, etc. Cases of infeasibility, unboundedness will not trigger
            this flag.

    """

    y : Optional[np.ndarray] = None

    optimal    = False
    infeasible = False
    unbounded  = False
    inf_or_unb = False
    inaccurate = False
    error      = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.infeasible or self.unbounded:
            self.inf_or_unb = True

        # Since SDPResult objects are created internally by backends,
        # the following exceptions are used to ensure the __init__ method is called correctly.
        # The exceptions are not expected to be raised (or caught) in user code.
        if self.optimal and self.inf_or_unb:
            raise ValueError("Status optimal and inf_or_unb cannot be True at the same time.")
        if self.error and (self.optimal or self.inf_or_unb):
            raise ValueError("Status error and (optimal or inf_or_unb) cannot be True at the same time.")
        if self.optimal and self.y is None:
            raise ValueError("Status optimal requires y to be not None.")
        if not (self.y is None or isinstance(self.y, np.ndarray)):
            raise TypeError(f"y must be None or np.ndarray, but got {type(self.y)}.")

    def __str__(self):
        return f"SDPResult(optimal={self.optimal}, infeasible={self.infeasible}, unbounded={self.unbounded}, inf_or_unb={self.inf_or_unb}, inaccurate={self.inaccurate}, error={self.error})"
    def __repr__(self):
        return self.__str__()

    def __array__(self):
        if self.y is None:
            raise ValueError("y is None.")
        return self.y
    def _sympify_(self):
        if self.y is None:
            raise ValueError("y is None.")
        from ..arithmetic import rep_matrix_from_numpy
        return rep_matrix_from_numpy(self.y)

    @property
    def success(self) -> bool:
        return self.optimal and (not self.inaccurate)
    def raises(self):
        if not self.success:
            pass
        return self.y


# The current expected status does not support combinations of status,
# e.g. optimal and inaccurate.
_EXPECTED_STATUS = {
    'optimal': True,
    'infeasible': False,
    'unbounded': False,
    'inf_or_unb': False,
    'inaccurate': False,
    'error': False,
}

class SDPError(Exception):
    """Base class for exceptions in this module."""
    result = None
    def __init__(self, result: SDPResult):   
        self.result = result
        error_messages = []
        for key, value in _EXPECTED_STATUS.items():
            v = getattr(result, key)
            if v != value:
                error_messages.append(f"{key}={getattr(result, key)}")
        message = ", ".join(error_messages)
        super().__init__(f"SDP solution failed: {message}")

    @property
    def y(self) -> Optional[np.ndarray]:
        return self.result.y

    @property
    def optimal(self) -> bool:
        return self.result.optimal
    @property
    def infeasible(self) -> bool:
        return self.result.infeasible
    @property
    def unbounded(self) -> bool:
        return self.result.unbounded
    @property
    def inf_or_unb(self) -> bool:
        return self.result.inf_or_unb
    @property
    def inaccurate(self) -> bool:
        return self.result.inaccurate
    @property
    def error(self) -> bool:
        return self.result.error

    @classmethod
    def from_kwargs(cls, **kwargs):
        result = SDPResult(**kwargs)
        if result.success:
            raise ValueError("SDPError.from_kwargs should be called with failed result.")
        return cls(result)



# class SDPProblemError(SDPError):
#     """Raised when the SDP is infeasible or unbounded."""

# class SDPInfeasibleError(SDPProblemError):
#     """Raised when the SDP is infeasible."""

# class SDPUnboundedError(SDPProblemError):
#     """Raised when the objective is unbounded."""

# class NoInstanceMeta(type):
#     def __call__(cls, *args, **kwargs):
#         raise TypeError(f"Cannot instantiate {cls.__name__!r} class")

# class SDPStatusClass(metaclass=NoInstanceMeta):
#     name       = 'status'
#     optimal    = False
#     error      = False
#     inf_or_unb = False
#     infeasible = False
#     unbounded  = False
#     error_cls  = None

# class StatusOptimal(SDPStatusClass):
#     name       = 'optimal'
#     optimal    = True

# class StatusInfOrUnb(SDPStatusClass):
#     name       = 'infeasible_or_unbounded'
#     inf_or_unb = True
#     error_cls  = SDPProblemError

# class StatusInfeasible(SDPStatusClass):
#     name       = 'infeasible'
#     inf_or_unb = True
#     infeasible = True
#     error_cls  = SDPInfeasibleError

# class StatusUnbounded(SDPStatusClass):
#     name       = 'unbounded'
#     inf_or_unb = True
#     unbounded  = True
#     error_cls  = SDPUnboundedError

# class StatusError(SDPStatusClass):
#     name       = 'error'
#     error      = True
#     error_cls  = SDPError

# class SDPStatus(metaclass=NoInstanceMeta):
#     """Namespace for SDP statuses."""
#     OPTIMAL = StatusOptimal
#     INFEASIBLE_OR_UNBOUNDED = StatusInfOrUnb
#     INFEASIBLE = StatusInfeasible
#     UNBOUNDED = StatusUnbounded
#     ERROR = StatusError


class SolverConfigs:
    verbose = 0
    max_iters = 200
    tol_gap_abs = 1e-8
    tol_gap_rel = 1e-8
    tol_fsb_abs = 1e-8
    tol_fsb_rel = 1e-8
    solver_options = dict()
    _KEYS = ('verbose', 'max_iters', 'tol_gap_abs', 'tol_gap_rel', 'tol_fsb_abs', 'tol_fsb_rel', 'solver_options')
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