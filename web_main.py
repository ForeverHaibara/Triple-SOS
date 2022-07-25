from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sum_of_square import *
from sos_manager import *
import sympy as sp 

_debug_ = False 

class SOS_WEB(Flask):
    def __init__(self, name):
        super().__init__(name)
        self.SOS_Manager : SOS_Manager = SOS_Manager()

app = SOS_WEB(__name__)
CORS(app)

@app.route('/process/preprocess', methods=['POST'])
def preprocess():
    app.SOS_Manager.setPoly(request.get_json()['poly'], cancel = True)
    poly = app.SOS_Manager.poly
    if poly is None:
        return 

    if app.SOS_Manager.poly_isfrac:
        cancel = app.SOS_Manager.getStandardForm()
    else:
        cancel = ''

    n = deg(poly)
    coeffs = poly.coeffs()
    monoms = app.SOS_Manager.std_monoms
    monoms.append((-1,-1,0))  # tail flag

    t = 0
    txts = [] 
    for i in range(n+1):
        for j in range(i+1):
            if monoms[t][0] == n - i and monoms[t][1] == i - j:
                if isinstance(coeffs[t],sp.core.numbers.Float):
                    txt = f'{round(float(coeffs[t]),4)}'
                else:
                    txt = f'{coeffs[t].p}' + (f'/{coeffs[t].q}' if coeffs[t].q != 1 else '')
                    if len(txt) > 10:
                        txt = f'{round(float(coeffs[t]),4)}'
                t += 1
            else:
                txt = '0'
            txts.append(txt)
            
    # restore
    monoms.pop()

    return jsonify(n = n, txts = txts, heatmap = app.SOS_Manager.grid_val,
                    cancel = cancel)


@app.route('/process/sos', methods=['POST'])
def SumOfSquare():
    app.SOS_Manager.tangents = [tg 
                        for tg in request.get_json()['tangents'].split('\n') if len(tg) > 0]
    result = app.SOS_Manager.GUI_SOS(request.get_json()['poly'],
                                     skip_setpoly=True, skip_findroots=True, skip_tangents=True,
                                     verbose_updeg=True)
    
    return jsonify(latex = app.SOS_Manager.sosresults[0], 
                    txt  = app.SOS_Manager.sosresults[1],
                    formatted = app.SOS_Manager.sosresults[2],
                    success = (len(result) > 0))


@app.route('/process/rootangents', methods=['POST'])
def RootsAndTangents():
    rootsinfo = app.SOS_Manager.GUI_findRoot()
    tangents = app.SOS_Manager.GUI_getTangents()
    return jsonify(rootsinfo = rootsinfo, tangents = tangents)


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