import re

import numpy as np
import gradio as gr
from PIL import Image

from src.utils import RootsInfo
from src.utils.text_process import short_constant_parser
from src.gui.sos_manager import SOS_Manager

def gradio_markdown() -> gr.Markdown:
    version = gr.__version__
    if version >= '4.0':
        return gr.Markdown(latex_delimiters=[{"left": "$", "right": "$", "display": False}])
    return gr.Markdown()

def _convert_to_gradio_latex(content):
    version = gr.__version__
    replacement = {
        '\n': ' \n\n ',
        '$$': '$',
        '&': ' ',
        '\\\\': ' $ \n\n  $',
        '\\begin{aligned}': '',
        '\\end{aligned}': '',
        '\\left\\{': '',
        '\\right.': '',
    }
    if version >= '4.0':
        replacement['\\frac'] = '\\dfrac'

    for k,v in replacement.items():
        content = content.replace(k,v)

    if version >= '4.0':
        content = re.sub('\$(.*?)\$', '$\\ \\1$', content, flags=re.DOTALL)
    else:
        content = re.sub('\$(.*?)\$', '$\\\displaystyle \\1$', content, flags=re.DOTALL)
    return content

DEPLOY_CONFIGS = {
    'LinearSOS': {
        'verbose': False,
    },
    'StructuralSOS': {
    },
    'SymmetricSOS': {

    },
    'SDPSOS': {
        'verbose': False,
        'degree_limit': 12,
    }
}

# DEPLOY_CONFIGS = {}

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
                                outputs = [self.image, self.coefficient_triangle])
            self.compute_btn.click(fn = self.solve, inputs = [self.input_box, self.methods_btn],
                                outputs = [self.output_box, self.output_box_latex, self.output_box_txt, self.output_box_formatted, 
                                            self.image, self.coefficient_triangle,
                                            self.output_box2])

    def set_poly(self, input_text, with_poly = False):
        res = SOS_Manager.set_poly(input_text)
        if res is None or res.get('poly') is None:
            return {self.image: None, self.coefficient_triangle: None}

        def _render_heatmap(res):
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
            res2['grid'] = res['grid']
        return res2


    def solve(self, input_text, methods):
        # self.SOS_Manager.set_poly(input_text, cancel = True)
        res0 = self.set_poly(input_text, with_poly = True)
        solution = None
        poly = res0.pop('poly', None)
        grid = res0.pop('grid', None)
        if poly is not None:
            if 'Linear' in methods:
                rootsinfo = SOS_Manager.findroot(poly, grid, verbose = False)
            else:
                rootsinfo = RootsInfo()
            try:
                solution = SOS_Manager.sum_of_square(
                    poly,
                    rootsinfo = rootsinfo,
                    method_order = ['%sSOS'%method for method in methods],
                    configs = DEPLOY_CONFIGS
                )
            except Exception as e:
                pass

        if solution is not None:
            gradio_latex = _convert_to_gradio_latex(solution.str_latex)
            res = {
                self.output_box: gradio_latex,
                self.output_box2: gradio_latex,
                self.output_box_latex: solution.str_latex,
                self.output_box_txt: solution.str_txt,
                self.output_box_formatted: solution.str_formatted,
            }
        else:
            res = {
                self.output_box: 'No solution found.',
                self.output_box2: 'No solution found.',
                self.output_box_latex: 'No solution found.',
                self.output_box_txt: 'No solution found.',
                self.output_box_formatted: 'No solution found.',
            }
            if poly is not None and poly.domain.is_Numerical\
                   and (not poly.is_zero) and poly.is_homogeneous and len(methods) == 4:
                print(input_text)
        res.update(res0)
        return res


if __name__ == '__main__':
    interface = GradioInterface()
    interface.demo.launch(show_error=True) #, debug=True)
