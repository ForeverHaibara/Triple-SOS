import sympy as sp
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# from sum_of_square import *
from src.utils.expression.solution import SolutionSimple
from src.gui.sos_manager import SOS_Manager

_debug_ = False

class SOS_WEB(Flask):
    def __init__(self, name):
        super().__init__(name)
        self.SOS_Manager : SOS_Manager = SOS_Manager()

app = SOS_WEB(__name__)
CORS(app)

@app.route('/process/preprocess', methods=['POST'])
def preprocess():
    app.SOS_Manager.set_poly(request.get_json()['poly'], cancel = True)
    poly = app.SOS_Manager.poly
    
    # if poly is None:
    #     return

    if app.SOS_Manager._poly_info['isfrac']:
        cancel = app.SOS_Manager.get_standard_form()
    else:
        cancel = ''

    n = app.SOS_Manager.deg
    coeffs = poly.coeffs()
    monoms = poly.monoms()
    monoms.append((-1,-1,0))  # tail flag
    
    t = 0
    txts = []
    for i in range(n+1):
        for j in range(i+1):
            if monoms[t][0] == n - i and monoms[t][1] == i - j:
                if isinstance(coeffs[t], sp.core.numbers.Float):
                    txt = f'{round(float(coeffs[t]),4)}'
                else:
                    v = coeffs[t].as_numer_denom()
                    txt = f'{v[0]}' + (f'/{v[1]}' if v[1] != 1 else '')
                    if len(txt) > 10:
                        txt = f'{round(float(coeffs[t]),4)}'
                t += 1
            else:
                txt = '0'
            txts.append(txt)
            
    # restore
    monoms.pop()

    factor = ''
    print(request.get_json()['factor'])
    if request.get_json()['factor'] == True:
        factor = app.SOS_Manager.get_standard_form(formatt = 'factor')


    return jsonify(n = n, txts = txts, heatmap = app.SOS_Manager.grid.grid_color,
                    cancel = cancel, factor = factor)


@app.route('/process/sos', methods=['POST'])
def SumOfSquare():
    req = request.get_json()
    # app.SOS_Manager._roots_info['tangents'] = [tg
    #                     for tg in req['tangents'].split('\n') if len(tg) > 0]

    solution = app.SOS_Manager.sum_of_square()
                                    #  use_structural_method=req['use_structural_method'])
    if solution is None:
        return jsonify(latex = '', txt = '', formatted = '', success = False)

    if isinstance(solution, SolutionSimple):
        # remove the aligned environment
        latex_ = '$$%s$$'%solution.str_latex[17:-15].replace('&','')

    return jsonify(latex = latex_,
                    txt  = solution.str_txt,
                    formatted = solution.str_formatted,
                    success = True)


@app.route('/process/rootangents', methods=['POST'])
def RootsAndTangents():
    rootsinfo = app.SOS_Manager.findroot()
    tangents = [str(_) for _ in rootsinfo.tangents]
    return jsonify(rootsinfo = rootsinfo.gui_description, tangents = tangents)


@app.route('/')
def hello_world():
    return render_template('triples.html')





if __name__ == '__main__':
    if _debug_:
        # python "D:\Python Projects\Trials\Inequalities\Triples\web_main.py" dev
        from flask_script import Manager
        manager = Manager(app)
        @manager.command
        def dev():
            from livereload import Server
            live_server = Server(app.wsgi_app)
            live_server.watch(
                "D:\\Python Projects\\Trials\\Inequalities\\Triples\\*.*"
            )
            live_server.serve(open_url_delay=True)
        manager.run()
    else:
        app.run(port=5000)
