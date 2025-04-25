import numpy as np
import sympy as sp

from ..caller import (_DUAL_BACKENDS, solve_numerical_dual_sdp, solve_numerical_primal_sdp,
    DualBackendCVXOPT, DualBackendCLARABEL
)
from ..settings import SDPError

SOLVERS = _DUAL_BACKENDS
SOLVERS = {'cvxopt': DualBackendCVXOPT}

class SDPDualProblems:
    """
    Each of the problem should return
    (x0_and_space, objective, constraints), answer
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def problem_sdpa(cls):
        # https://sdpa-python.github.io/docs/usage/
        x0_and_space = {0: (np.zeros((4,)), np.eye(4))}
        objective = [11,0,0,-23]
        A, b = [[-10,-4,-4,0],[0,0,0,8],[0,8,8,2]], [-48,8,-20]
        constraints = [
            (A, b, '<'), # or equality cones
            ([0,-1,1,0], 0, '=') # x[1] == x[2]
        ]
        answer = 11*5.9 - 23*1.0 # 41.9
        return (x0_and_space, objective, constraints), answer

    @classmethod
    def problem_sdplib(cls):
        # https://github.com/vsdp/SDPLIB
        x0_and_space = {0: (-np.diag([1,2,3,4]).flatten(),
            sp.Matrix([[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,5,2,0,0,2,6]]).T)}
        objective = [10,20]
        answer = 30.
        return (x0_and_space, objective, []), answer

    @classmethod
    def problem_truss1(cls):
        # https://github.com/vsdp/SDPLIB/blob/master/data/truss1.dat-s
        x0_and_space = dict(enumerate([
            ([0,0,0,0],[[0,0,0,0,0,-1],[0,0,0,0,0,0],[0,0,0,0,0,0],[-1,0,0,0,0,0]]),
            ([0,0,0,0],[[0,0,0,0,0,-1],[0,-1,0,0,0,0],[0,-1,0,0,0,0],[-1,0,0,0,0,0]]),
            ([0,0,0,0],[[0,0,0,0,0,-1],[0,0,0,-0.5,-0.5,0],[0,0,0,-0.5,-0.5,0],[-1,0,0,0,0,0]]),
            ([0,0,0,0],[[0,0,0,0,0,-1],[0,0,0,-1,0,0],[0,0,0,-1,0,0],[-1,0,0,0,0,0]]),
            ([0,0,0,0],[[0,0,0,0,0,-1],[0,-0.5,0.5,0,0,0],[0,-0.5,0.5,0,0,0],[-1,0,0,0,0,0]]),
            ([0,0,0,0],[[0,0,0,0,0,-1],[0,0,1,0,-1,0],[0,0,1,0,-1,0],[-1,0,0,0,0,0]]),
            ([1],[[0,0,0,0,0,1]])
        ]))
        objective = [-1,0,-2,0,0,0]
        answer = -9.
        return (x0_and_space, objective, []), answer

    @classmethod
    def problem_cvxopt(cls):
        x0_and_space = {
            'G1': (np.array(([[33., -9.], [-9., 26.]])).flatten(),
                -np.array(([[-7., -11., -11., 3.],
                            [ 7., -18., -18., 8.],
                            [-2.,  -8.,  -8., 1.]])).T),
            'G2': (np.array([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]).flatten(),
                -np.array([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
                        [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
                        [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  8., 6.]]).T)
        }
        return (x0_and_space, [1,-1,1], []), -3.15354500191818 # a root of a degree-18 polynomial


    @classmethod
    def problem_dof1(cls):
        # A = [[2-y, 1], [1, -y]], min(-y)
        x0_and_space = {0: ([2,1,1,0], [[-1],[0],[0],[-1]])}
        return (x0_and_space, sp.Matrix([-1]), []), 2**.5 - 1


    @classmethod
    def problem_infeasible(cls):
        # A = [[2, 2*a, 2*a - 8], [2*a, 2, 1 - a], [2*a - 8, 1 - a, 2]]
        # A >> 0 infeasible because [1,-2,1].T @ A @ [1,-2,1] < 0
        x0_and_space = {'A': ([2,0,-8,0,2,1,-8,1,2], np.array([[0,2,2,2,0,-1,2,-1,0]]).T)}
        return (x0_and_space, [0], []), None

    @classmethod
    def problem_infeasible2(cls):
        # A = [[1, 1+a], [1+a, 1]], B = [[b, a], [a, .3]]], a + b = 2, 2*a + b >= 1, -a >= 1
        x0_and_space = {'A': ([1,1,1,1],np.array([[0,1,1,0],[0,0,0,0]]).T),
                        'B': ([0,0,0,.3],np.array([[0,1,1,0],[1,0,0,0]]).T)}
        constraints = [([1,1.],2,'='), ([[2,1],[-1,0]],[1,1],'>=')]
        return (x0_and_space, [0,0], constraints), None

    @classmethod
    def problem_unbounded(cls):
        # A = [[x+y,y+1], [y+1,x]], min (x+2y), s.t. -3x-5y<=2
        x0_and_space = {0: ([0,1,1,0], np.array([[1,0,0,1],[1,1,1,0]]).T.tolist())}
        return (x0_and_space, [1,2], [([-3,-5],2,'<=')]), -np.inf


    @classmethod
    def problem_empty0(cls):
        x0_and_space = dict()
        return (x0_and_space, [], []), 0.

    @classmethod
    def problem_empty1(cls):
        x0_and_space = {0: (np.zeros((4,),dtype=float), np.zeros((4,0)))}
        return (x0_and_space, [], []), 0.

    @classmethod
    def problem_empty2(cls):
        x0_and_space = {1: (np.array([1,-2,-2,4])/3, np.zeros((4,0))),
                        2: (np.array([3.14]), np.zeros((1,0)))}
        constraints = [(np.zeros((0,)), -2/3, '<'),
                        (np.zeros((4,0)), np.zeros((4,)), '=='),
                        (np.zeros((2,0)), [2e5, 3.14e-6], '<')]
        return (x0_and_space, [], constraints), None

    @classmethod
    def problem_empty_numerical(cls):
        x0_and_space = {1: (np.array([1,-2,-2,4])/3, np.zeros((4,0))),
                        2: (np.array([3.14]), np.zeros((1,0)))}
        constraints = [(np.zeros((0,)), 1e-15, '=='),
                        (np.zeros((2,0)), [-1.2e-14, 1.123e-15], '='),
                        (np.zeros((2,0)), [1e-14, -2], '>')]
        return (x0_and_space, [], constraints), 0.

    @classmethod
    def problem_empty3(cls):
        x0_and_space = {1: (np.array([7,5,5,4])/6, np.zeros((4,0))),
                        2: (np.full((9,), -1.2), np.zeros((9,0)))}
        return (x0_and_space, [], []), None

    @classmethod
    def problem_empty4(cls):
        x0_and_space = {0: (np.array([7,-51,-51,372])/6, np.zeros((4,0)))}
        constraints = [(np.zeros((1,0)), 0.5, '>'), (np.zeros((2,0)), [0.2,-1e-2], '>')]
        return (x0_and_space, [], constraints), None


class SDPPrimalProblems:
    """
    Each of the problem should return
    (x0_and_space, objective, constraints), answer
    """
    @classmethod
    def collect(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.startswith('problem')}

    @classmethod
    def from_dual(cls, x0_and_space, objective, constraints, answer):
        """
        Get a new problem from the dual form by Lagrangian dual.
        Users should ensure strong duality holds.
        """
        if constraints:
            raise NotImplementedError('Lagrangian dual is not implemented for constraints.')

        new_x0 = objective
        new_obj = []
        new_space = dict()
        for key, (x0, space) in x0_and_space.items():
            new_obj.append(np.array(x0).astype(float).flatten())
            new_space[key] = np.array(space).astype(float).T
        new_obj = np.concatenate(new_obj)
        return ((new_x0, new_space), new_obj, []), -answer

    @classmethod
    def problem_sdplib(cls):
        args, answer = SDPDualProblems.problem_sdplib()
        return cls.from_dual(*args, answer)

    @classmethod
    def problem_truss1(cls):
        args, answer = SDPDualProblems.problem_truss1()
        return cls.from_dual(*args, answer)

    @classmethod
    def problem_cvxopt(cls):
        args, answer = SDPDualProblems.problem_cvxopt()
        return cls.from_dual(*args, answer)

    @classmethod
    def problem_dof1(cls):
        return (([5], {0: [2]}), [-3], []), -7.5


    @classmethod
    def problem_empty0(cls):
        x0_and_space = ([], dict())
        return (x0_and_space, [], []), 0.

    @classmethod
    def problem_empty1(cls):
        x0_and_space = ([], dict())
        return (x0_and_space, [], 
                    [(np.ones((2,0)), [1,-1e-15], '<'),
                    (np.ones((1,0)), [1e-15], '=')]), 0.

    @classmethod
    def problem_empty2(cls):
        x0_and_space = ([], dict())
        return (x0_and_space, [], 
                    [(np.ones((2,0)), [1,-1e-15], '<'),
                    (np.ones((1,0)), [1e-2], '=')]), None
  
    @classmethod
    def problem_empty3(cls):
        x0_and_space = ([], dict())
        return (x0_and_space, [], 
                    [(np.ones((2,0)), [1,-1e-15], '<'),
                    (np.ones((1,0)), [1e-15], '='),
                    (np.ones((2,0)), np.array([-0.8,2.4]).reshape(2,1), '>')]), None

    @classmethod
    def problem_infeasible(cls):
        # Todd 2001
        # X[0,0] = 0 but X[1,0] = X[0,1] = 1 -> infeasible
        # add an extra constraint to prevent solvers from claiming early termination
        x0_and_space = ([0,1], {0:[[2,0,0,0],[0,0.5,0.5,0]]})
        return (x0_and_space, [1,1,1,1], [([0,-2,-2,3.6], 100., '<')]), None

    @classmethod
    def problem_unbounded(cls):
        # [[x, y], [y, 1-x-4y]] >> 0, [[x-y]] >> 0, 3x+y<-2
        x0_and_space = ([1,0], {0:[[1,2,2,1],[1,-0.5,-0.5,0]], 1:[[0],[-1.]]})
        return (x0_and_space, [1,1,1,1,0], [([4,0,0,0,-1], [-2], '<')]), np.inf


def _test_dual_or_primal(solvers, problems, solve_func, func_name=""):
    for solver_name, solver in solvers.items():
        if not solver.is_available():
            continue
        for problem_name, problem in problems.items():
            # print(f'{problem_name} with {solver_name}')
            def safe_solve_func(*args, **kwargs):
                try:
                    return solve_func(*args, **kwargs)
                except Exception as e:
                    e.args = (f'{func_name} solver {solver_name} failed on problem {problem_name} by raising an error {e}',) + e.args
                    raise e
            (x0_and_space, objective, constraints), answer = problem()
            if answer is not None and not np.isinf(answer):
                y = safe_solve_func(x0_and_space, objective, constraints,
                            solver=solver_name, return_result=False)
                objective = np.array(objective).astype(float).flatten()
                assert isinstance(y, np.ndarray) and y.shape == objective.shape,\
                    f'{func_name} solver {solver_name} failed on problem {problem_name} with unexpected return {y}.'

                obj = np.dot(objective, y)
                assert abs(obj - answer) / (abs(answer) + 1) < 1e-6,\
                    f'{func_name} solver {solver_name} failed on problem {problem_name} with answer {y}, ' + \
                    f'objective {obj} != {answer}, error {abs(obj - answer) / (abs(answer) + 1)}.'
            elif answer is None:
                y = safe_solve_func(x0_and_space, objective, constraints,
                        solver=solver_name, return_result=True)
                # assert False, f'Dual solver {solver_name} returned {y} on INFEASIBLE problem {problem_name}.'
                assert y.inf_or_unb and (not y.unbounded),\
                    f'{func_name} solver {solver_name} claimed {y} on INFEASIBLE problem {problem_name}.'
            elif np.isinf(answer):
                y = safe_solve_func(x0_and_space, objective, constraints,
                        solver=solver_name, return_result=True)
                # assert False, f'Dual solver {solver_name} returned {y} on UNBOUNDED problem {problem_name}.'
                assert y.inf_or_unb or (not y.infeasible),\
                    f'{func_name} solver {solver_name} claimed {y} on UNBOUNDED problem {problem_name}.'

def test_duals():
    solvers = SOLVERS
    problems = SDPDualProblems.collect()
    _test_dual_or_primal(solvers, problems, solve_numerical_dual_sdp, func_name="Dual")

def test_primal():
    solvers = SOLVERS
    problems = SDPPrimalProblems.collect()
    _test_dual_or_primal(solvers, problems, solve_numerical_primal_sdp, func_name="Primal")