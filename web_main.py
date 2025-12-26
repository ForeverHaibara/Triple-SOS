from typing import Tuple, List, Dict, Union, Optional

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room
from flask_cors import CORS

from sympy import Poly, Expr, Symbol, Rational
from sympy.combinatorics import PermutationGroup

from triples.utils.text_process import pl, coefficient_triangle_latex
from triples.utils import (
    optimize_poly, Root
)
from triples.core import Solution
from triples.gui.sos_manager import SOSManager

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


@app.route('/process/preprocess', methods=['POST'])
def preprocess():
    """
    Process the input polynomial and emit the result to the client.
    The parameters are passed as a JSON object in the request body.

    Parameters
    ----------
    sid : str
        The session ID.
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
    if len(set(gens)) != len(gens):
        raise ValueError("Duplicate generators are not allowed.")
    if perm.degree != len(gens):
        raise ValueError("The degree of the permutation group"\
                    + "must be equal to the number of generators.")

    parser_configs = req["parser_configs"]
    parser = SOSManager.make_parser(
        gens, perm,
        **{key: parser_configs[key] for key in [
            "lowercase",
            "cyclic_sum_func",
            "cyclic_prod_func",
            "preserve_patterns",
            "scientific_notation",
        ] if key in parser_configs
        }
    )
    req["parser"] = parser

    # Step 1. convert to sympy.Expr
    raw_expr = None
    try:
        raw_expr = parser(req["poly"], return_type="expr",
                parse_expr_kwargs = {"evaluate": False})
    except Exception:
        pass
    if raw_expr is None:
        return jsonify()

    # Step 2. try to convert to a fraction
    expr = (None, None)
    try:
        # This only expands the expr to fraction of polynomials,
        # and there is no text processing.
        expr = pl(raw_expr, gens, perm, return_type="frac")
    except:
        pass
    if expr is None or (isinstance(expr, tuple) and expr[0] is None):
        expr = raw_expr

    # Step 3. apply transformations as required
    # return the new text and new expression
    expr, text = SOSManager.apply_transformations(
        expr, gens, perm,
        **{key: parser_configs[key] for key in [
            "cancel",
            "homogenize",
            "dehomogenize",
            "standardize_text",
            "cyclic_sum_func",
            "cyclic_prod_func",
            "omit_mul",
            "omit_pow",
        ] if key in parser_configs
        }
    )

    # Step 4. render the coeff triangle and the heatmap
    degree, triangle, heatmap = SOSManager.render_coeff_triangle_and_heatmap(raw_expr, expr)

    sid = req.pop("sid")

    req["expr"] = expr
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
        roots = _findroots(sid,
            kwargs["expr"],
            timestamp = kwargs.get("timestamp", 0),
        )
        # Do not push the roots to the args of sum_of_squares,
        # since the roots are found without constraints.

    if not ("sos" in actions):
        return

    solution = None
    try:
        ineq_constraints = SOSManager.parse_constraints_dict(
            kwargs["ineq_constraints"], kwargs["parser"])
        eq_constraints = SOSManager.parse_constraints_dict(
            kwargs["eq_constraints"], kwargs["parser"])

        if "sos" in actions:
            sos_configs = kwargs.get("sos_configs", {})
            solution = _sum_of_squares(sid,
                kwargs["expr"],
                ineq_constraints,
                eq_constraints,
                gens = kwargs["gens"],
                symmetry = kwargs["perm"],
                timestamp = kwargs.get("timestamp", 0),
                **{key: sos_configs[key] for key in [
                    "methods",
                    "time_limit",
                    "configs",
                ] if key in sos_configs
                }
            )
    except Exception:
        pass
    finally:
        if solution is None:
            # tell the client that the SOS terminated
            socketio.emit(
                "sos", {
                    "latex": "", 
                    "txt": "", 
                    "formatted": "", 
                    "success": False,
                    "timestamp": kwargs.get("timestamp", 0),
                },
                to=sid
            )


def _findroots(
    sid: int,
    expr: Union[Poly, Tuple[Poly, Poly]],
    *,
    timestamp: int = 0,
):
    poly = expr
    roots = []
    if isinstance(poly, Poly) and (poly.domain.is_ZZ or poly.domain.is_QQ):
        roots = optimize_poly(poly, [], [poly])
    socketio.emit(
        'findroots', {
            'rootsinfo': [
                [str(v if isinstance(v, Rational) else v.n(8)) for v in r]
                    for r in roots
            ],
            'timestamp': timestamp,
        },
        to=sid
    )
    return roots


def _sum_of_squares(
    sid: int,
    expr: Expr,
    ineq_constraints: Dict[Expr, Expr],
    eq_constraints: Dict[Expr, Expr],
    *,
    methods: Optional[List[str]] = None,
    time_limit: float = 300.,
    configs: dict = {},
    gens: Tuple[Symbol, ...],
    symmetry: PermutationGroup,
    timestamp: int = 0,
):
    solution = SOSManager.sum_of_squares(
        expr,
        ineq_constraints,
        eq_constraints,
        methods = methods,
        time_limit = time_limit,
        configs = configs
    )

    if solution is None:
        return None

    solution = solution.rewrite_symmetry(gens, symmetry)

    # # remove the aligned environment
    tex = solution.to_string(mode='latex', lhs_expr=Symbol('\\text{LHS}'),
        together=True, cancel=True, settings={'long_frac_ratio':2})#.replace('aligned', 'align*')

    socketio.emit(
        "sos", {
            "latex": tex, 
            "txt": solution.to_string(mode="txt", lhs_expr=Symbol('LHS')),
            "formatted": solution.to_string(mode="formatted", lhs_expr=Symbol('LHS')),
            "success": True,
            "timestamp": timestamp,
        },
        to=sid
    )

    return solution


@app.route('/process/latexcoeffs', methods=['POST'])
def get_latex_coeffs():
    req = request.get_json()
    parser_configs = req["parser_configs"]
    parser = SOSManager.make_parser(req["gens"], req["perm"],
        **{key: parser_configs[key] for key in [
            "lowercase",
            "cyclic_sum_func",
            "cyclic_prod_func",
            "preserve_patterns",
            "scientific_notation",
        ] if key in parser_configs
        }
    )
    poly = parser(req["poly"], return_type='frac')
    if poly is None or poly[0] is None:
        return None
    coeffs = coefficient_triangle_latex(poly[0],
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
        SOSManager.time_limit = 300.0

    print('=' * 50)
    print('Running the server at http://%s:%d'%(HOST, PORT))    
    print('=' * 50)
    socketio.run(app, host=HOST, port=PORT, debug=False)
