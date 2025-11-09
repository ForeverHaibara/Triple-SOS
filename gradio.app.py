import re

import numpy as np
from sympy import Function, Symbol
from sympy.external.importtools import version_tuple
import gradio as gr
from PIL import Image

from triples.utils.text_process import short_constant_parser
from triples.core import sum_of_squares
from triples.gui.sos_manager import SOS_Manager
from triples.gui.linebreak import recursive_latex_auto_linebreak

GRADIO_VERSION = tuple(version_tuple(gr.__version__))
# https://github.com/gradio-app/gradio/pull/8822
GRADIO_LATEX_SUPPORTS_ALIGNED = (GRADIO_VERSION[0] == 4 and GRADIO_VERSION[1] in (39, 40, 41, 43, 44))\
    or GRADIO_VERSION[0] > 4

def gradio_markdown() -> gr.Markdown:
    version = GRADIO_VERSION
    if version >= (4, 0):
        return gr.Markdown(latex_delimiters=[{"left": "$", "right": "$", "display": False}])
    return gr.Markdown()

def _convert_to_gradio_latex(content):
    version = GRADIO_VERSION
    replacement = {
        '\n': ' \n\n ',
        '$$': '$',
        # '\\left\\{': '',
        # '\\right.': '',
    }
    if not GRADIO_LATEX_SUPPORTS_ALIGNED:
        replacement.update({
            '&': ' ',
            '\\\\': ' $ \n\n  $',
            '\\begin{aligned}': '',
            '\\end{aligned}': '',
        })
    if version >= (4, 0):
        replacement['\\frac'] = '\\dfrac'
    # else:
    #     replacement['\\begin{aligned}'] = ''
    #     replacement['\\end{aligned}'] = ''

    for k,v in replacement.items():
        content = content.replace(k,v)

    if version >= (4, 0):
        content = re.sub('\$(.*?)\$', '$\\ \\1$', content, flags=re.DOTALL)
    else:
        content = re.sub('\$(.*?)\$', '$\\\displaystyle \\1$', content, flags=re.DOTALL)
    return content


class GradioInterface():
    def __init__(self):

        with gr.Blocks(css="#coefficient_triangle{height: 600px;margin-top:-5px}") as self.demo:
            # input and output blocks
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    self.input_box = gr.Textbox(label="Input", placeholder="Input a homogeneous, cyclic three-var polynomial like s(a^2)^2-3s(a^3b)", show_label=False, max_lines=1)
                    self.coefficient_triangle = gr.HTML('<div id="coeffs" style="width: 100%; height: 600px; position: absolute;"></div>', elem_id="coefficient_triangle")

                with gr.Column(scale=1):#, variant='compact'):
                    # an image display block (note: gr.Image makes low resolution images, use gr.HTML for better quality)
                    self.image = gr.Image(show_label = False, height=280, width=280)
                    # self.image = gr.HTML('<div id="heatmap" style="width=300; height=300; position:absolute;"></div>', elem_id="heatmap")
                    self.compute_btn = gr.Button("Compute", scale=1, variant="primary")

                    self.methods_btn = gr.CheckboxGroup(
                        ['Structural', 'Linear', 'Symmetric', 'SDP'],
                        value = ['Structural', 'Linear', 'Symmetric', 'SDP'],
                        label = "Methods"
                    )

                    with gr.Tabs():
                        with gr.Tab("Display"):
                            self.output_box = gradio_markdown()
                        with gr.Tab("LaTeX"):
                            self.output_box_latex = gr.TextArea(label="Result", show_copy_button=True)
                        with gr.Tab("txt"):
                            self.output_box_txt = gr.TextArea(label="Result", show_copy_button=True)
                        with gr.Tab("formatted"):
                            self.output_box_formatted = gr.TextArea(label="Result", show_copy_button=True)

            self.output_box2 = gradio_markdown()
            # add a link to the github repo
            self.external_link = gr.HTML('<a href="https://github.com/ForeverHaibara/Triple-SOS">GitHub Link</a>')

            self.input_box.submit(fn = self.set_poly, inputs = [self.input_box], 
                                outputs = [self.image, self.coefficient_triangle], show_api=False)
            self.compute_btn.click(fn = self.solve, inputs = [self.input_box, self.methods_btn],
                                outputs = [self.output_box, self.output_box_latex, self.output_box_txt, self.output_box_formatted, 
                                            self.image, self.coefficient_triangle,
                                            self.output_box2], show_api=False)

            with gr.Row(render=False, equal_height=False):
                # Add hidden components for API
                self.api_input_expr = gr.Textbox(visible=False, min_width=0, label="Problem")
                self.api_input_ineq = gr.Textbox(visible=False, min_width=0, label="Ineqs",
                    info="Nonnegative expressions separated by ';', e.g., a;b;c for a>=0, b>=0, c>=0.")
                self.api_input_eq = gr.Textbox(visible=False, min_width=0, label="Eqs",
                    info="Zero expressions separated by ';', e.g., a*b*c-1;x^2+y^2-1 for a*b*c-1=0,x^2+y^2-1=0.")
                self.api_output = gr.JSON(visible=False, min_width=0, height=0)

                # API endpoint
                self.api_input_expr.change(
                    fn=self.api_sum_of_squares,
                    inputs=[self.api_input_expr, self.api_input_ineq, self.api_input_eq],
                    outputs=[self.api_output],
                    api_name="sum_of_squares"
                )


    def set_poly(self, input_text, with_poly = False):
        res = SOS_Manager.set_poly(input_text)
        if res is None or res.get('poly') is None:
            return {self.image: None, self.coefficient_triangle: None}

        def _render_heatmap(res):
            if res.get('grid') is None:
                return None
            image = res['grid'].save_heatmap(backgroundcolor=255)
            image = np.vstack([np.full((8, image.shape[1], 3), 255, np.uint8), image])
            side_white = np.full((image.shape[0], 4, 3), 255, np.uint8)
            image = np.hstack([side_white, image, side_white])
            image = Image.fromarray(image).resize((300,300), Image.LANCZOS)
            return image

        # def _render_heatmap():
        #     # html = '<div id="heatmap" style="width: 100%; height: 100%; top: 0; left: 0; position:absolute;"></div>'
        #     return html


        # render coefficient triangle
        def _render_coefficient_triangle(res):
            html = '<div id="coeffs" style="width: 100%; height: 600px; position: absolute;">'

            n = res['degree']
            coeffs = res['poly'].coeffs()
            monoms = res['poly'].monoms()
            monoms.append((-1,-1,0))  # tail flag
            
            l = 100. / (1 + n)
            ulx = (50 - l * n / 2)
            uly = (29 - 13 * n * l / 45)
            fontsize = max(11,int(28-1.5*n))
            lengthscale = .28 * fontsize / 20.;
            t = 0
            txts = []
            title_parser = lambda chr, deg: '' if deg <= 0 else (chr if deg == 1 else chr + str(deg))
            for i in range(n+1):
                for j in range(i+1):
                    if monoms[t][0] == n - i and monoms[t][1] == i - j:
                        txt = short_constant_parser(coeffs[t])
                        t += 1
                    else:
                        txt = '0'
                    x = (ulx + l*(2*i-j)/2 - len(txt) * lengthscale - 2*int(len(txt)/4))
                    y = (uly + l*j*13/15)
                    title = title_parser('a',n-i) + title_parser('b',i-j) + title_parser('c',j)
                    txt = '<p style="position: absolute; left: %f%%; top: %f%%; font-size: %dpx; color: %s;" title="%s">%s</p>'%(
                        x, y, fontsize, 'black' if txt != '0' else 'rgb(180,180,180)', title, txt
                    )
                    txts.append(txt)
            html = html + '\n'.join(txts) + '</div>'
            return html

        image = _render_heatmap(res)
        html = _render_coefficient_triangle(res)

        res2 = {self.image: image, self.coefficient_triangle: html}
        if with_poly:
            res2['poly'] = res['poly']
            res2['grid'] = res.get('grid', None)
        return res2


    def solve(self, input_text, methods):
        # self.SOS_Manager.set_poly(input_text, cancel = True)
        res0 = self.set_poly(input_text, with_poly = True)
        solution = None
        poly = res0.pop('poly', None)
        grid = res0.pop('grid', None)
        if poly is not None:
            try:
                ineq_constraints = poly.free_symbols if SOS_Manager.CONFIG_ALLOW_NONSTANDARD_GENS else poly.gens
                solution = SOS_Manager.sum_of_squares(
                    poly,
                    ineq_constraints = list(ineq_constraints),
                    eq_constraints = [],
                    methods = ['%sSOS'%method for method in methods] + ['Pivoting', 'Reparametrization'],
                )
            except Exception as e:
                # print(e)
                pass

        gens = poly.free_symbols if SOS_Manager.CONFIG_ALLOW_NONSTANDARD_GENS else poly.gens
        gens = sorted(gens, key=lambda x:x.name)
        lhs_expr = Function('F')(*gens) if len(gens) > 0 else Symbol('\\text{LHS}')
        if solution is not None:
            tex = solution.to_string(mode='latex', lhs_expr=lhs_expr, settings={'long_frac_ratio': 2})
            if GRADIO_LATEX_SUPPORTS_ALIGNED:
                tex = recursive_latex_auto_linebreak(tex)
            tex = '$$%s$$' % tex
            gradio_latex = _convert_to_gradio_latex(tex)
            res = {
                self.output_box: gradio_latex,
                self.output_box2: gradio_latex,
                self.output_box_latex: tex,
                self.output_box_txt: solution.to_string(mode='txt', lhs_expr=lhs_expr),
                self.output_box_formatted: solution.to_string(mode='formatted', lhs_expr=lhs_expr),
            }
        else:
            res = {
                self.output_box: 'No solution found.',
                self.output_box2: 'No solution found.',
                self.output_box_latex: 'No solution found.',
                self.output_box_txt: 'No solution found.',
                self.output_box_formatted: 'No solution found.',
            }
            # if poly is not None and poly.domain.is_Numerical\
            #        and (not poly.is_zero) and poly.is_homogeneous and len(methods) == 4:
            if poly is not None and len(methods) == 4 and poly.total_degree() > 2:
                print(input_text)
        res.update(res0)
        return res

    def api_sum_of_squares(self, expr, ineq_constraints, eq_constraints):
        # from sympy.parsing.sympy_parser import parse_expr
        # from sympy.parsing.sympy_parser import T
        from sympy import sympify
        # parser = lambda x: parse_expr(x, transformations=T[:7])
        def parser(x):
            x = sympify(x)
            if any(len(s.name) > 1 for s in x.free_symbols):
                # prevent a*b miswritten as ab
                raise ValueError("Variable names should be single characters.")
            return x
        parser_t = lambda x: tuple(map(parser, (x.split(':', 1) if ':' in x else (x, x))))
        try:
            # Parse string inputs to SymPy expressions
            parsed_expr = parser(expr)
            parsed_ineqs = dict(parser_t(c) for c in (ineq_constraints or '').split(';') if c)
            parsed_eqs = dict(parser_t(c) for c in (eq_constraints or '').split(';') if c)

            # Call the core function
            result = sum_of_squares(
                poly=parsed_expr,
                ineq_constraints=parsed_ineqs,
                eq_constraints=parsed_eqs
            )
            tex = None
            tex_aligned = None
            if result is not None:
                lhs_expr = Symbol('\\text{LHS}')
                tex = result.to_string(mode='latex', lhs_expr=lhs_expr, settings={'long_frac_ratio': 2})
                tex_aligned = recursive_latex_auto_linebreak(tex)

            return {
                'success': bool(result is not None),
                'solution': result.solution.doit() if result is not None else None,
                'latex': tex,
                'latex_aligned': tex_aligned,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'solution': None,
                'latex': None,
                'latex_aligned': None,
                'error': str(e)
            }


if __name__ == '__main__':
    SOS_Manager.verbose = False

    interface = GradioInterface()

    ALLOW_CORS = True
    if ALLOW_CORS:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = interface.demo.app
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "https://foreverhaibara.github.io", 
                # "http://localhost:5173",             # Vite 开发环境
                # "http://127.0.0.1:5173"              # 本地测试
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"]
        )
    
    interface.demo.launch(show_error=True) #, debug=True)