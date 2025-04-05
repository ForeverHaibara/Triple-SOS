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

class SDPStatus:
    """Namespace for SDP statuses."""
    OPTIMAL = StatusOptimal
    INFEASIBLE_OR_UNBOUNDED = StatusInfOrUnb
    INFEASIBLE = StatusInfeasible
    UNBOUNDED = StatusUnbounded
    ERROR = StatusError