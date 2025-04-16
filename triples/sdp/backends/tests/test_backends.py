import numpy as np
import sympy as sp

from ..caller import _DUAL_BACKENDS, solve_numerical_dual_sdp
from ..settings import SDPError

class SDPDualProblems:
    """
    Each of the problem should return
    (x0_and_space, objective, constraints), answer
    """
    def collect(self):
        return {k: getattr(self, k) for k in dir(self) if k.startswith('problem')}

    def problem_sdpa(self):
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

    def problem_sdplib(self):
        # https://github.com/vsdp/SDPLIB
        x0_and_space = {0: (-np.diag([1,2,3,4]).flatten(),
            sp.Matrix([[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,5,2,0,0,2,6]]).T)}
        objective = [10,20]
        answer = 30.
        return (x0_and_space, objective, []), answer

    def problem_truss1(self):
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

    def problem_dof1(self):
        # A = [[2-y, 1], [1, -y]], min(-y)
        x0_and_space = {0: ([2,1,1,0], [[-1],[0],[0],[-1]])}
        return (x0_and_space, sp.Matrix([-1]), []), 2**.5 - 1


    def problem_infeasible(self):
        # A = [[2, 2*a, 2*a - 8], [2*a, 2, 1 - a], [2*a - 8, 1 - a, 2]]
        # A >> 0 infeasible because [1,-2,1].T @ A @ [1,-2,1] < 0
        x0_and_space = {'A': ([2,0,-8,0,2,1,-8,1,2], np.array([[0,2,2,2,0,-1,2,-1,0]]).T)}
        return (x0_and_space, [0], []), None

    def problem_infeasible2(self):
        # A = [[1, 1+a], [1+a, 1]], B = [[b, a], [a, .3]]], a + b = 2, 2*a + b >= 1, -a >= 1
        x0_and_space = {'A': ([1,1,1,1],np.array([[0,1,1,0],[0,0,0,0]]).T),
                        'B': ([0,0,0,.3],np.array([[0,1,1,0],[1,0,0,0]]).T)}
        constraints = [([1,1.],2,'='), ([[2,1],[-1,0]],[1,1],'>=')]
        return (x0_and_space, [0,0], constraints), None

    def problem_unbounded(self):
        # A = [[x+y,y+1], [y+1,x]], min (x+2y), s.t. -3x-5y<=2
        x0_and_space = {0: ([0,1,1,0], np.array([[1,0,0,1],[1,1,1,0]]).T.tolist())}
        return (x0_and_space, [1,2], [([-3,-5],2,'<=')]), -np.inf


    def problem_empty0(self):
        x0_and_space = dict()
        return (x0_and_space, [], []), 0.

    def problem_empty1(self):
        x0_and_space = {0: (np.zeros((4,),dtype=float), np.zeros((4,0)))}
        return (x0_and_space, [], []), 0.

    def problem_empty2(self):
        x0_and_space = {1: (np.array([1,-2,-2,4])/3, np.zeros((4,0))),
                        2: (np.array([3.14]), np.zeros((1,0)))}
        constraints = [(np.zeros((0,)), -2/3, '<'),
                        (np.zeros((4,0)), np.zeros((4,)), '=='),
                        (np.zeros((2,0)), [2e5, 3.14e-6], '<')]
        return (x0_and_space, [], constraints), None

    def problem_empty_numerical(self):
        x0_and_space = {1: (np.array([1,-2,-2,4])/3, np.zeros((4,0))),
                        2: (np.array([3.14]), np.zeros((1,0)))}
        constraints = [(np.zeros((0,)), 1e-15, '=='),
                        (np.zeros((2,0)), [-1.2e-14, 1.123e-15], '='),
                        (np.zeros((2,0)), [1e-14, -2], '>')]
        return (x0_and_space, [], constraints), 0.

    def problem_empty3(self):
        x0_and_space = {1: (np.array([7,5,5,4])/6, np.zeros((4,0))),
                        2: (np.full((9,), -1.2), np.zeros((9,0)))}
        return (x0_and_space, [], []), None

    def problem_empty4(self):
        x0_and_space = {0: (np.array([7,-51,-51,372])/6, np.zeros((4,0)))}
        constraints = [(np.zeros((1,0)), 0.5, '>'), (np.zeros((2,0)), [0.2,-1e-2], '>')]
        return (x0_and_space, [], constraints), None


def test_duals():
    for solver_name, solver in _DUAL_BACKENDS.items():
        if not solver.is_available():
            continue
        problems = SDPDualProblems().collect()
        for problem_name, problem in problems.items():
            # print(f'{problem_name} with {solver_name}')
            (x0_and_space, objective, constraints), answer = problem()
            if answer is not None and not np.isinf(answer):
                y = solve_numerical_dual_sdp(x0_and_space, objective, constraints,
                            solver=solver_name, return_result=False)
                objective = np.array(objective).astype(float).flatten()
                assert isinstance(y, np.ndarray) and y.shape == objective.shape,\
                    f'Dual solver {solver_name} failed on problem {problem_name} with unexpected return {y}.'

                obj = np.dot(objective, y)
                assert abs(obj - answer) / (abs(answer) + 1) < 1e-6,\
                    f'Dual solver {solver_name} failed on problem {problem_name} with answer {y}, ' + \
                    f'objective {obj} != {answer}, error {abs(obj - answer) / (abs(answer) + 1)}.'
            elif answer is None:
                y = solve_numerical_dual_sdp(x0_and_space, objective, constraints,
                        solver=solver_name, return_result=True)
                # assert False, f'Dual solver {solver_name} returned {y} on INFEASIBLE problem {problem_name}.'
                assert y.inf_or_unb and (not y.unbounded),\
                    f'Dual solver {solver_name} claimed {y} on INFEASIBLE problem {problem_name}.'
            elif np.isinf(answer):
                y = solve_numerical_dual_sdp(x0_and_space, objective, constraints,
                        solver=solver_name, return_result=True)
                # assert False, f'Dual solver {solver_name} returned {y} on UNBOUNDED problem {problem_name}.'
                assert y.inf_or_unb or (not y.infeasible),\
                    f'Dual solver {solver_name} claimed {y} on UNBOUNDED problem {problem_name}.'