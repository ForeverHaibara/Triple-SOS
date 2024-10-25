# import inspect
# import ctypes
# import threading

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room
from flask_cors import CORS

import sympy as sp

from src.utils.roots import RootTangent, RootsInfo
from src.utils.text_process import pl
from src.utils.expression.solution import SolutionSimple
from src.gui.sos_manager import SOS_Manager

# def _async_raise(tid, exctype):
#     '''Raises an exception in the threads with id tid'''
#     if not inspect.isclass(exctype):
#         raise TypeError("Only types can be raised (not instances)")
#     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
#                                                      ctypes.py_object(exctype))
#     if res == 0:
#         raise ValueError("invalid thread id")
#     elif res != 1:
#         # "if it returns a number greater than one, you're in trouble,
#         # and you should call it again with exc=NULL to revert the effect"
#         ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
#         raise SystemError("PyThreadState_SetAsyncExc failed")

# class ThreadWithExc(threading.Thread):
#     '''A thread class that supports raising an exception in the thread from
#        another thread.
#     '''
#     def _get_my_tid(self):
#         """determines this (self's) thread id

#         CAREFUL: this function is executed in the context of the caller
#         thread, to get the identity of the thread represented by this
#         instance.
#         """
#         if not self.is_alive(): # Note: self.isAlive() on older version of Python
#             raise threading.ThreadError("the thread is not active")

#         # do we have it cached?
#         if hasattr(self, "_thread_id"):
#             return self._thread_id

#         # no, look for it in the _active dict
#         for tid, tobj in threading._active.items():
#             if tobj is self:
#                 self._thread_id = tid
#                 return tid

#         # TODO: in python 2.6, there's a simpler way to do: self.ident

#         raise AssertionError("could not determine the thread's id")

#     def raise_exc(self, exctype):
#         """Raises the given exception type in the context of this thread.

#         If the thread is busy in a system call (time.sleep(),
#         socket.accept(), ...), the exception is simply ignored.

#         If you are sure that your exception should terminate the thread,
#         one way to ensure that it works is:

#             t = ThreadWithExc( ... )
#             ...
#             t.raise_exc( SomeException )
#             while t.isAlive():
#                 time.sleep( 0.1 )
#                 t.raise_exc( SomeException )

#         If the exception is to be caught by the thread, you need a way to
#         check that your thread has caught it.

#         CAREFUL: this function is executed in the context of the
#         caller thread, to raise an exception in the context of the
#         thread represented by this instance.
#         """
#         _async_raise( self._get_my_tid(), exctype )

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

    gens = req.get('gens', 'abc')
    gens = tuple(sp.Symbol(_) for _ in gens)
    perm = SOS_Manager._parse_perm_group(req.get('perm'))
    req['gens'] = gens
    req['perm'] = perm

    result = SOS_Manager.set_poly(
        req['poly'],
        gens = gens,
        perm = perm,
        render_triangle = True,
        render_grid = True,
        factor = req.get('factor', False)
    )
    if result is None:
        return jsonify()

    n = result['degree']
    txt = result['txt']
    triangle = result['triangle']
    grid = result.get('grid', None)
    grid_color = grid.grid_color if grid is not None else None
 
    sid = req.pop('sid')
    req.update(result)
    socketio.start_background_task(findroot, sid, **req)
    # thread = ThreadWithExc(target=findroot, args=(sid,), kwargs=req)
    # thread.daemon = True
    # thread.start()
    # socketio.sleep(3)
    # thread.raise_exc(SystemExit)

    return jsonify(n = n, txt = txt, triangle = triangle, heatmap = grid_color)

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
            tangents = [_.as_factor_form(remove_minus_sign=True, perm=kwargs['perm']) for _ in rootsinfo.tangents]
            socketio.emit(
                'rootangents',
                {
                    'rootsinfo': rootsinfo.gui_description,
                    'tangents': tangents,
                    'timestamp': kwargs.get('timestamp', 0)
                },
                to=sid
            )
        elif 'sos' in kwargs['actions']:
            tangents = []
            for tg in kwargs['tangents'].split('\n'):
                if len(tg) > 0:
                    try:
                        tg = pl(tg, gens=kwargs['gens'], perm=kwargs['perm'])
                        if tg is not None:
                            # and (tg.domain in (sp.ZZ, sp.QQ)):
                            tg = tg.as_expr()
                            # tangents.append(RootTangent(tg))
                            if 'configs' not in kwargs:
                                kwargs['configs'] = {'LinearSOS': {'tangents': []}}
                            elif 'LinearSOS' not in kwargs['configs']:
                                kwargs['configs']['LinearSOS'] = {'tangents': []}
                            elif 'tangents' not in kwargs['configs']['LinearSOS']:
                                kwargs['configs']['LinearSOS']['tangents'] = []
                            kwargs['configs']['LinearSOS']['tangents'].append(tg**2)
                    except:
                        pass
            rootsinfo.tangents = tangents
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
            gens = kwargs['gens'],
            perm = kwargs['perm'],
            rootsinfo = rootsinfo,
            method_order = method_order,
            configs = kwargs['configs']
        )

        assert solution is not None, 'No solution found.'
    except Exception as e:
        # raise e
        return socketio.emit(
            'sos',
            {'latex': '', 'txt': '', 'formatted': '', 'success': False,
                'timestamp': kwargs.get('timestamp', 0)},
            to=sid
        )

    if isinstance(solution, SolutionSimple):
        # # remove the aligned environment
        # latex_ = '$$%s$$'%solution.str_latex[17:-15].replace('&','')
        latex_ = solution.str_latex#.replace('aligned', 'align*')

    return socketio.emit(
        'sos',
        {'latex': latex_, 'txt': solution.str_txt, 'formatted': solution.str_formatted, 'success': True,
            'timestamp': kwargs.get('timestamp', 0)},
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
    poly = req['poly']
    gens = tuple(sp.Symbol(_) for _ in req.get('gens', 'abc'))
    perm = SOS_Manager._parse_perm_group(req.get('perm'))
    coeffs = SOS_Manager.latex_coeffs(poly, gens, perm,
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
        SOS_Manager.verbose = False

    print('=' * 50)
    print('Running the server at http://%s:%d'%(HOST, PORT))    
    print('=' * 50)
    socketio.run(app, host=HOST, port=PORT, debug=False)