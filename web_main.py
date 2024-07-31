from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room
from flask_cors import CORS

import sympy as sp

from src.utils.roots import RootTangent, RootsInfo
from src.utils.text_process import pl
from src.utils.expression.solution import SolutionSimple
from src.gui.sos_manager import SOS_Manager


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
    poly : str
        The input polynomial.
    factor : bool
        Whether to factor the polynomial. If True, the factor form is returned as the text.
    actions : list[str]
        Additional actions to perform.

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

    result = SOS_Manager.set_poly(
        req['poly'],
        render_triangle = True,
        render_grid = True,
        factor = req.get('factor', False)
    )
    if result is None:
        return jsonify()

    n = result['degree']
    txt = result['txt']
    triangle = result['triangle']
    grid = result['grid']
 
    sid = req.pop('sid')
    req.update(result)
    socketio.start_background_task(findroot, sid, **req)

    return jsonify(n = n, txt = txt, triangle = triangle, heatmap = grid.grid_color)

def findroot(sid, **kwargs):
    """
    Find the root information of the polynomial and emit the result to the client.

    Parameters
    ----------
    sid : str
        The session ID.
    poly : str
        The input polynomial.
    grid : Grid
        The grid of the polynomial coefficients. This is passed in
        internally by the `preprocess` function.
    tangents : list[str]
        The tangents to the roots. If provided, the tangents are not recalculated.
    actions : list[str]
        Additional actions to perform.

    Returns
    ----------
    rootsinfo : str
        The string representation of the roots information.
    tangents : list[str]
        The tangents to the roots.    
    """
    if 'findroot' in kwargs['actions']:
        poly = kwargs['poly']
        grid = kwargs['grid']
        rootsinfo = SOS_Manager.findroot(poly, grid, verbose=False)
        tangents = kwargs.get('tangents')
        if tangents is None:
            tangents = [_.as_factor_form(remove_minus_sign=True) for _ in rootsinfo.tangents]
            socketio.emit(
                'rootangents',
                {'rootsinfo': rootsinfo.gui_description, 'tangents': tangents}, to=sid
            )
        elif 'sos' in kwargs['actions']:
            tangents = []
            for tg in kwargs['tangents'].split('\n'):
                if len(tg) > 0:
                    try:
                        tg = pl(tg)
                        if tg is not None and (tg.domain in (sp.ZZ, sp.QQ)):
                            tg = tg.as_expr()
                            tangents.append(RootTangent(tg))
                    except:
                        pass
    if 'sos' in kwargs['actions']:
        kwargs['rootsinfo'] = rootsinfo
        sum_of_square(sid, **kwargs)


def sum_of_square(sid, **kwargs):
    """
    Perform the sum of square decomposition, and emit the result to the client.
    Always emit the result to the client, even if the solution is None or an error occurs.

    Parameters
    ----------
    sid : str
        The session ID.
    poly : str
        The input polynomial.
    methods : dict[str, bool]
        The methods to use.
    configs : dict[str, dict]
        The configurations for each method.
    rootsinfo : RootsInfo
        The roots information. This is passed in internally by the `findroot` function.

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
    rootsinfo = kwargs['rootsinfo'] or RootsInfo()
    try:
        method_order = [key for key, value in kwargs['methods'].items() if value]

        solution = SOS_Manager.sum_of_square(
            kwargs['poly'],
            rootsinfo = rootsinfo,
            method_order = method_order,
            configs = kwargs['configs']
        )

        assert solution is not None, 'No solution found.'
    except:
        return socketio.emit(
            'sos', {'latex': '', 'txt': '', 'formatted': '', 'success': False}, to=sid
        )

    if isinstance(solution, SolutionSimple):
        # # remove the aligned environment
        # latex_ = '$$%s$$'%solution.str_latex[17:-15].replace('&','')
        latex_ = solution.str_latex#.replace('aligned', 'align*')

    return socketio.emit(
        'sos',
        {'latex': latex_, 'txt': solution.str_txt, 'formatted': solution.str_formatted, 'success': True},
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

    Returns
    ----------
    coeffs : str
        The LaTeX representation of the coefficients
    """
    poly = request.get_json()['poly']
    coeffs = SOS_Manager.latex_coeffs(poly, tabular = True, document = True, zeros='\\textcolor{lightgray}')
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
        SOS_Manager.verbose = False

    print('=' * 50)
    print('Running the server at http://%s:%d'%(HOST, PORT))    
    print('=' * 50)
    socketio.run(app, host=HOST, port=PORT, debug=False)