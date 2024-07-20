from logging import getLogger

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

from src.utils.roots import RootTangent, RootsInfo
from src.utils.text_process import pl
from src.utils.expression.solution import SolutionSimple
from src.gui.sos_manager import SOS_Manager


class SOS_WEB(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

app = SOS_WEB(__name__, template_folder = './')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# getLogger('socketio').disabled = True

@socketio.on('connect')
def on_connect():
    sid = request.sid
    join_room(sid)
    print('Joined', sid)

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    leave_room(sid)
    print('Left', sid)

@app.route('/process/preprocess', methods=['POST'])
def preprocess():
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
    if 'sos' in kwargs['actions']:
        kwargs['rootsinfo'] = rootsinfo
        sum_of_square(sid, **kwargs)


def sum_of_square(sid, **kwargs):
    """
    Perform the sum of square decomposition, and emit the result to the client.
    Always emit the result to the client, even if the solution is None or an error occurs.
    """
    rootsinfo = kwargs['rootsinfo'] or RootsInfo()
    try:
        rootsinfo.tangents = [
            RootTangent(pl(tg).as_expr()) for tg in kwargs['tangents'].split('\n') if len(tg) > 0
        ]

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
    coeffs = SOS_Manager.latex_coeffs(tabular = True, document = True)
    return jsonify(coeffs = coeffs)

@app.route('/')
def index():
    # user_id = str(uuid4())
    return render_template('triples.html') #, user_id = user_id)


def gevent_launch(app):
    # https://flask-socketio.readthedocs.io/en/latest/deployment.html
    from gevent import monkey
    from flask_socketio import SocketIO
    monkey.patch_all()
    socketio = SocketIO(app)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    DEPLOY = True #False
    # if not DEPLOY:
    socketio.run(app, port=5000)