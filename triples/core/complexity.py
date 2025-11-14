class ProblemComplexity:
    """
    time  : `E(time)` expected time to solve the problem.
    prob  : `E(success prob)` assuming the problem is correct.
    length: `E(length of solution | success)` expected length of the solution if the problem is solved.
    status: Status code or timestamp when evaluated.

    #### Comparison of ProblemComplexity
    Consider solving a problem by two methods, A and B, with time t1, t2 and success probability p1, p2.
    There are two choices: 1. try A first, and if it fails, try B; 2. try B first, and if it fails, try A.
    The expected time cost using A->B is `t1 + t2(1 - p1)`, and the expected time cost using B->A is `t2 + t1(1 - p2)`.
    Then:

        `t1 + t2(1 - p1) < t2 + t1(1 - p2)     <=>     t1/p1 < t2/p2`
    """
    EPS = 1e-14
    time: float = 0
    prob: float = 0
    length: float = 0
    status: int = 0
    def __init__(self, time, prob, length=0., status=0):
        self.time = time
        self.prob = prob
        self.length = length
        self.status = status

    def __str__(self) -> str:
        return f"{{time: {self.time:.2f}, prob: {self.prob:.2f}, length: {self.length:.2f}}}"

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'ProblemComplexity':
        return ProblemComplexity(self.time, self.prob, self.length, self.status)

    def __and__(a, b) -> 'ProblemComplexity':
        return ProblemComplexity(
            a.time + b.time, a.prob * b.prob, a.length + b.length
        )

    def __or__(a, b) -> 'ProblemComplexity':
        p = 1 - (1 - a.prob) * (1 - b.prob)
        return ProblemComplexity(
            a.time + b.time * (1 - a.prob), p,
            (a.length * a.prob + b.length * (1 - a.prob) * b.prob)/max(p, ProblemComplexity.EPS)
        )

    def __gt__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) >  b.time / max(b.prob, ProblemComplexity.EPS)
    def __lt__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) <  b.time / max(b.prob, ProblemComplexity.EPS)
    def __ge__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) >= b.time / max(b.prob, ProblemComplexity.EPS)
    def __le__(a, b) -> bool:
        return a.time / max(a.prob, ProblemComplexity.EPS) <= b.time / max(b.prob, ProblemComplexity.EPS)

    def __eq__(a, b) -> bool:
        # todo: is it well-defined?
        return a.time == b.time and a.prob == b.prob and a.length == b.length

