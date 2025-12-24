from typing import Tuple, Optional

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room
from flask_cors import CORS

from sympy import Poly, Symbol, Rational, sympify

from triples.utils.text_process import (
    pl, poly_get_factor_form, poly_get_standard_form, degree_of_expr,
    coefficient_triangle,
)
from triples.utils import (
    optimize_poly, Root
)
from triples.core import Solution
from triples.gui.grid import GridRender
from triples.gui.sos_manager import SOSManager
from triples.gui.linebreak import recursive_latex_auto_linebreak

class SOS_WEB(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

app = SOS_WEB(__name__, template_folder = './')
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def on_connect():
    sid = request.sid
    join_room(sid)
    print('Joined Room', sid)

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    leave_room(sid)
    print('Left Room', sid)


def apply_transformations(
    expr,
    gens,
    perm_group,
    *,
    cancel = True,
    homogenize = False,
    dehomogenize = None,
    standardize_text = None,
    omit_mul = True,
    omit_pow = True,
) -> Tuple[Poly, str]:

    if cancel:
        if isinstance(expr, tuple):
            # discard the denominator if required
            if expr[1] != Poly(1, expr[1].gens, domain=expr[1].domain) \
                    and standardize_text is None:
                # the denominator is not one, and the expression is modified
                # -> recompute the expression
                standardize_text = "sort"
            expr = expr[0]

    if isinstance(expr, Poly):
        poly = expr
        if homogenize and (not poly.is_homogeneous) and poly.degree(len(gens)-1) == 0:
            gen = poly.gens[-1]
            poly = poly.eval(gen, 0).homogenize(gen)
            if standardize_text is None:
                standardize_text = "sort"

        if dehomogenize is not None and dehomogenize is not False:
            try:
                dehomogenize_val = sympify(dehomogenize)
                if len(dehomogenize_val.free_symbols) == 0:  # dehomogenize is a constant
                    for i in range(len(poly.gens) - 1, -1, -1):
                        if poly.degree(i) != 0:
                            poly = poly.eval(poly.gens[i], dehomogenize_val)
                            poly = poly.as_poly(*gens, domain=poly.domain)
                            if standardize_text is None:
                                standardize_text = "sort"
                            break
            except Exception:
                # Ignore dehomogenization errors
                pass
        expr = poly

    elif isinstance(expr, tuple):
        pass

    text = None
    if standardize_text:
        if standardize_text == "factor":
            text = poly_get_factor_form(poly,
                perm_group, omit_mul=omit_mul, omit_pow=omit_pow)
        elif standardize_text == "sort":
            text = poly_get_standard_form(poly,
                perm_group, omit_mul=omit_mul, omit_pow=omit_pow)
        elif standardize_text == "expand":
            text = poly_get_standard_form(poly,
                "trivial", omit_mul=omit_mul, omit_pow=omit_pow)

    return expr, text


def render_coeff_triangle_and_heatmap(
    raw_expr,
    expr,
) -> Tuple[Optional[int], Optional[list], Optional[list]]:
    if isinstance(expr, tuple):
        # discard the denominator if required
        expr = expr[0]
    if not isinstance(expr, Poly):
        return None, None, None

    poly = expr
    degree = poly.total_degree()
    if poly.is_zero:
        degree = degree_of_expr(raw_expr, poly.gens)

    heatmap = None
    triangle = None
    if len(poly.gens) == 3 or len(poly.gens) == 4:
        triangle = coefficient_triangle(poly, degree)
        if poly.domain.is_Numerical: 
            size = 60 if len(poly.gens) == 3 else 18
            grid = GridRender.render(poly, size=size, with_color=True)
            heatmap = grid.grid_color if grid is not None else None

    return degree, triangle, heatmap


@app.route('/process/preprocess', methods=['POST'])
def preprocess():
    """
    Process the input polynomial and emit the result to the client.
    The parameters are passed as a JSON object in the request body.

    Parameters
    ----------
    sid : str
        The session ID.
    poly : str
        The input polynomial.
    standardize_text : str
        The standardization of the text. If None, the text is not standardized.
        See `SOSManager.set_poly` for more information.
    actions : list[str]
        Additional actions to perform. Supporting 

    Returns
    ----------
    n : int
        The degree of the polynomial.
    txt : str
        The formatted polynomial.
    triangle : str
        The coefficients formatted as a triangle.
    heatmap : list[tuple[int, int, int, int]]
        The heatmap of the polynomial coefficients in RGBA format.
    """
    req = request.get_json()
    req_input = req["poly"]
    if len(req_input.strip()) == 0:
        return jsonify()

    gens = tuple(Symbol(_) for _ in req["gens"])
    perm = SOSManager.parse_perm_group(req["perm"])
    req["gens"] = gens
    req["perm"] = perm

    # Step 1. convert to sympy.Expr
    raw_expr = None
    try:
        raw_expr = pl(req["poly"], gens, perm, return_type="text",
            parse_expr_kwargs = {"evaluate": False})
    except Exception:
        pass
    if raw_expr is None:
        return jsonify()

    # Step 2. try to convert to fraction
    expr = (None, None)
    try:
        expr = pl(raw_expr, gens, perm, return_type="frac")
    except:
        pass
    if expr is None:
        expr = raw_expr

    # Step 3. apply transformations as required
    # return the new text and new expression
    expr, text = apply_transformations(
        expr, gens, perm,
        **{key: req[key] for key in [
            "cancel",
            "homogenize",
            "dehomogenize",
            "standardize_text",
            "omit_mul",
            "omit_pow"
        ] if key in req}
    )

    # Step 4. render the coeff triangle and the heatmap
    degree, triangle, heatmap = render_coeff_triangle_and_heatmap(raw_expr, expr)

    sid = req.pop("sid")

    req["poly"] = expr
    socketio.start_background_task(chained_actions, sid, **req)
    return jsonify(
        txt = text if text is not None else req_input,
        n = degree,
        triangle = triangle,
        heatmap = heatmap
    )


def chained_actions(sid, **kwargs):
    actions = kwargs.get("actions", [])
    if "findroot" in actions:
        roots = findroot(sid, **kwargs)
    if "sos" in actions:
        sum_of_squares(sid, **kwargs)


def findroot(sid, **kwargs):
    poly = kwargs['poly']
    roots = optimize_poly(poly, [], [poly]) \
        if poly.domain.is_ZZ or poly.domain.is_QQ else []
    socketio.emit(
        'rootangents',
        {
            'rootsinfo': 'Local Minima Approx:\n' + '\n'.join(
                [str(tuple(__ if isinstance(__, Rational) else __.n(8) for __ in r))
                        for r in roots[:min(len(roots), 5)]]),
            'tangents': [], # deprecated
            'timestamp': kwargs.get('timestamp', 0)
        },
        to=sid
    )


def sum_of_squares(sid, **kwargs):
    """
    Perform the sum of square decomposition, and emit the result to the client.
    Always emit the result to the client, even if the solution is None or an error occurs.

    Parameters
    ----------
    sid : str
        The session ID.
    poly : str
        The input polynomial.
    ineq_constraints: dict[str, str]
        The ineq constraints.
    eq_constraints: dict[str, str]
        The eq constraints.
    methods : dict[str, bool]
        The methods to use.
    configs : dict[str, dict]
        The configurations for each method.
    roots : list[Root]
        The roots of the polynomial.
    perm : PermutationGroup

    Returns
    ----------
    latex : str
        The LaTeX representation of the solution.
    txt : str
        The text representation of the solution.
    formatted : str
        The formatted representation of the solution.
    success : bool
        Whether the solution was found.
    """
    try:
        methods = [key for key, value in kwargs['methods'].items() if value]
        methods.extend(['Pivoting', 'Reparametrization'])

        gens = kwargs['gens']
        # ineq_constraints = kwargs['poly'].free_symbols if SOSManager.ALLOW_NONSTANDARD_GENS else gens

        def parse_constraint_dict(source):
            constraints = {}
            for key, value in source.items():
                key, value = key.strip(), value.strip()
                if len(key) == 0:
                    continue
                key = pl(key, gens, kwargs['perm'], return_type='expr')
                if len(value) != 0:
                    value = sympify(value)
                else:
                    value = key
                constraints[key] = value
            return constraints

        ineq_constraints = parse_constraint_dict(kwargs['ineq_constraints'])
        eq_constraints = parse_constraint_dict(kwargs['eq_constraints'])

        solution = SOSManager.sum_of_squares(
            kwargs['poly'],
            ineq_constraints = ineq_constraints,
            eq_constraints = eq_constraints,
            gens = gens,
            symmetry = kwargs['perm'],
            methods = methods,
            configs = kwargs['configs']
        )

        assert solution is not None, 'No solution found.'
    except Exception as e:
        return socketio.emit(
            "sos",
            {
                "latex": "", 
                "txt": "", 
                "formatted": "", 
                "success": False,
                "timestamp": kwargs.get("timestamp", 0)
            },
            to=sid
        )

    lhs_expr = Symbol('\\text{LHS}')
    if isinstance(solution, Solution):
        # # remove the aligned environment
        tex = solution.to_string(mode='latex', lhs_expr=lhs_expr,
            together=True, cancel=True, settings={'long_frac_ratio':2})#.replace('aligned', 'align*')
        tex = recursive_latex_auto_linebreak(tex)
        tex = '$$%s$$'%tex

    return socketio.emit(
        "sos",
        {
            "latex": tex, 
            "txt": solution.to_string(mode="txt", lhs_expr=lhs_expr),
            "formatted": solution.to_string(mode="formatted", lhs_expr=lhs_expr),
            "success": True,
            "timestamp": kwargs.get("timestamp", 0)
        },
        to=sid
    )


@app.route('/process/latexcoeffs', methods=['POST'])
def get_latex_coeffs():
    """
    Get the LaTeX representation of the coefficient triangle.

    Parameters
    ----------
    poly : str
        The input polynomial.
    gens : str
        The generator variables.
    perm : str
        The permutation group.

    Returns
    ----------
    coeffs : str
        The LaTeX representation of the coefficients
    """
    req = request.get_json()
    poly = req.get('poly', None)
    if poly is None:
        return None
    gens = tuple(Symbol(_) for _ in req.get('gens', 'abc'))
    perm = SOSManager.parse_perm_group(req.get('perm'))
    coeffs = SOSManager.latex_coeffs(poly, gens, perm,
                tabular = True, document = True, zeros='\\textcolor{lightgray}')
    return jsonify(coeffs = coeffs)


@app.route('/')
def index():
    return render_template('triples.html')


if __name__ == '__main__':
    # For deployment, please also configure the "host" variable in triples.html!!!
    DEPLOY = False

    HOST = '127.0.0.1'
    PORT = 5000
    if DEPLOY:
        HOST = '0.0.0.0'
        SOSManager.verbose = False

    print('=' * 50)
    print('Running the server at http://%s:%d'%(HOST, PORT))    
    print('=' * 50)
    socketio.run(app, host=HOST, port=PORT, debug=False)
