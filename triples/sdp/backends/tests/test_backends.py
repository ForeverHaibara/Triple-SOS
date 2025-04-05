import numpy as np
import sympy as sp

from ..caller import _DUAL_BACKENDS, solve_numerical_dual_sdp

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
            (A, b, '='), # equality cones
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

def test_duals():
    for solver_name in _DUAL_BACKENDS:
        problems = SDPDualProblems()
        for problem_name, problem in problems.collect().items():
            print(f'{problem_name} with {solver_name}')
            (x0_and_space, objective, constraints), answer = problem()
            if answer is not None and not np.isinf(answer):
                y = solve_numerical_dual_sdp(x0_and_space, objective, constraints, solver=solver_name)
                objective = np.array(objective).astype(float).flatten()
                assert isinstance(y, np.ndarray) and y.shape == objective.shape,\
                    f'Dual solver {solver_name} failed on problem {problem_name} with unexpected return {y}.'

                obj = np.dot(objective, y)
                assert abs(obj - answer) / (abs(answer) + 1) < 1e-6,\
                    f'Dual solver {solver_name} failed on problem {problem_name} with answer {y}, ' + \
                    f'objective {obj} != {answer}, error {abs(obj - answer) / (abs(answer) + 1)}.'