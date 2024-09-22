from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

from numpy import ndarray
from sympy import Matrix


class AbstractSDPProblem(ABC):
    """
    Abstract class for SDP problems, containing the signature of some basic methods.
    """
    _x0_and_space: Dict[str, Tuple[Matrix, Matrix]] = None

    @abstractmethod
    def _standardize_mat_dict(self, mat_dict: Dict[str, Matrix]) -> Dict[str, Matrix]: ...

    @abstractmethod
    def register_y(self, y: Union[Matrix, ndarray, Dict], perturb: bool = False, propagate_to_parent: bool = True) -> None: ...

    @property
    @abstractmethod
    def size(self) -> Dict[str, int]: ...