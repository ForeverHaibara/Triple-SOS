"""
Gradio Interface for Triple-SOS supporting deployment on online platforms.

Gradio LaTeX rendering is sensitive to the version, recommend >= 4.44 for a bug fix,
If 4.44 or >= 5 are not available, 3.44 is also recommended.
"""
import datetime
from functools import partial
from itertools import chain
from tokenize import TokenError
from typing import Tuple, List, Dict
import re

import numpy as np
from sympy import Poly, Expr, Symbol, sympify
from sympy.external.importtools import version_tuple
import gradio as gr
from PIL import Image

from triples.utils.text_process import pl, short_constant_parser
from triples.core import sum_of_squares
from triples.gui.sos_manager import SOSManager
from triples.gui.linebreak import recursive_latex_auto_linebreak

GRADIO_VERSION = tuple(version_tuple(gr.__version__))
# https://github.com/gradio-app/gradio/pull/8822
GRADIO_LATEX_SUPPORTS_ALIGNED = (GRADIO_VERSION[0] == 4 and GRADIO_VERSION[1] in (39, 40, 41, 43, 44))\
    or GRADIO_VERSION[0] > 4

LOCK_MARK = chr(128274)

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

def flatten_nested_dicts(dicts: List[dict]) -> Dict[tuple, list]:
    """
    >>> flatten_nested_dicts([{'a': 1, 'b': {'c': 2, 'd': 3}},
    ... {'a': 5, 'b': {'d': -1, 'c': 4}}]) # doctest:+SKIP
    {('a',): [1, 5], ('b', 'c'): [2, 4], ('b', 'd'): [3, -1]}
    """
    result = {}

    def traverse(nested_dict: dict, current_path: tuple = ()):
        for key, value in nested_dict.items():
            new_path = current_path + (key,)
            if isinstance(value, dict):
                traverse(value, new_path)
            else:
                if new_path not in result:
                    result[new_path] = []
                result[new_path].append(value)

    for d in dicts:
        traverse(d)
    return result


class GradioInterface():
    # Custom CSS for vertical layout styling
    css = """
    #coefficient_triangle{height: 600px;margin-top:-5px}
    .vertical-layout-center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .vertical-layout-container {
        max-width: 800px;
        width: 100%;
    }
    .constraint-row {
        margin-bottom: 8px;
    }
    .constraint-type {
        margin: 0 8px;
        align-self: center;
    }
    """

    description = f"""
    <hr>
    <div>
    GitHub Link: <a href="https://github.com/ForeverHaibara/Triple-SOS"
    >https://github.com/ForeverHaibara/Triple-SOS</a>
    <br><br>
    Deployment Date: {datetime.datetime.now().strftime("%Y-%m-%d")} &nbsp;&nbsp;
    Gradio Version: {gr.__version__}
    </div>
    """

    def __init__(self):
        self.layout = {
            "horizontal": {},
            "vertical": {},
        }
        with gr.Blocks(css=self.css) as self.demo:
            # State variables
            self.vertical_mode = gr.State(value=False)

            # Toggle button for layout mode
            with gr.Row():
                self.toggle_layout_btn = gr.Button(
                    "Toggle Vertical Layout",
                    variant="secondary",
                    size="sm"
                )
                gr.Markdown("", visible=False)  # Spacer to push toggle button to left

            # Main container that will change layout based on vertical_mode state
            with gr.Row(equal_height=False, visible=True) as self.horizontal_container:
                with gr.Column(scale=5):
                    self.input_box = gr.Textbox(label="Input", placeholder="Input a polynomial like s(a^2)^2-3s(a^3b)", show_label=False, max_lines=1)
                    self.coefficient_triangle = gr.HTML('<div id="coeffs" style="width: 100%; height: 600px; position: absolute;"></div>', elem_id="coefficient_triangle")

                    self._init_shared_outputs("horizontal")

                with gr.Column(scale=1) as self.right_column:
                    # an image display block (note: gr.Image makes low resolution images, use gr.HTML for better quality)
                    self.image = gr.Image(show_label = False, height=280, width=280)
                    # self.image = gr.HTML('<div id="heatmap" style="width=300; height=300; position:absolute;"></div>', elem_id="heatmap")
                    self.compute_btn = gr.Button("Compute", scale=1, variant="primary")

                    self._init_shared_components("horizontal")

            # Vertical layout container (initially hidden)
            with gr.Column("vertical-layout-center", visible=False) as self.vertical_container:
                with gr.Column("vertical-layout-container"):
                    # Input box in vertical layout
                    self.input_box_vertical = gr.Textbox(
                        label="Input",
                        placeholder="Input a polynomial like s(a^2)^2-3s(a^3b)",
                        show_label=False,
                        max_lines=20
                    )

                    # Right column content in vertical layout
                    with gr.Column():
                        self.compute_btn_vertical = gr.Button("Compute", variant="primary")

                        self._init_shared_components("vertical")
                        self._init_shared_outputs("vertical")

            # add a link to the github repo
            self.description_component = gr.HTML(self.description)

            # Organize components by layout for better code reuse and maintainability
            self.layout['horizontal'].update({
                'input': self.input_box,
                'compute_btn': self.compute_btn,
            })
            self.layout['vertical'].update({
                'input': self.input_box_vertical,
                'compute_btn': self.compute_btn_vertical,
            })


            Component = gr.components.Component
            flatten = flatten_nested_dicts(
                [self.layout['horizontal'], self.layout['vertical']]
            )
            flatten_values = list(chain.from_iterable(flatten.values()))
            flatten_values = [_ for _ in flatten_values if isinstance(_, Component)]

            # Toggle layout function
            self.toggle_layout_btn.click(
                fn=self.toggle_layout,
                inputs=[self.vertical_mode] + flatten_values,
                outputs=[
                    self.vertical_mode,
                    self.horizontal_container,
                    self.vertical_container
                ] + flatten_values,
                show_api=False
            )

            # Setup event handlers for both layouts
            self._setup_event_handlers()

            self._setup_api()

    def _init_shared_components(self, layout: str):
        # Generators section
        collection = self.layout[layout]

        collection['generators_input'] = gr.Textbox(
            label="Generators",
            placeholder="Enter lowercase letters (a-z), no duplicates",
            value="abc",
            max_lines=1
        )

        # Permutation section
        collection['perm_group'] = {}
        collection['perm_group']['radio'] = gr.Radio(
            ['Cyc', 'Sym', 'Custom'],
            label="Permutation",
            value="Cyc"
        )
        collection['perm_group']['input'] = gr.Textbox(
            label="Permutation Group",
            value="[[0,1,2],[1,2,0]]",
            max_lines=1,
            interactive=False
        )

        # Constraints section
        collection['constraints'] = {}
        with gr.Column():
            collection['constraints']['df'] = gr.Dataframe(
                value = [["a", "", "≥0"], ["b", "", "≥0"], ["c", "", "≥0"]],
                headers = ['Constraint', 'Alias', 'Type'],
                col_count = (3, "fixed"),
                datatype = "str",
                type = "array",
                show_fullscreen_button = False,
                label = "Constraints",
                pinned_columns = 3,
                static_columns = [2]
            )

            with gr.Row():
                kwargs = {"size": "sm"}
                collection['constraints']['add_inequality'] = gr.Button("+ Inequality", **kwargs)
                collection['constraints']['add_equality'] = gr.Button("+ Equality", **kwargs)
                collection['constraints']['positive_toggle'] = gr.Button(
                    f"Positive {LOCK_MARK}", variant="primary", **kwargs
                )
                collection['constraints']['clear'] = gr.Button("Clear", **kwargs)

        collection['methods_btn'] = gr.CheckboxGroup(
            ['Structural', 'Linear', 'Symmetric', 'SDP'],
            value = ['Structural', 'Linear', 'Symmetric', 'SDP'],
            label = "Methods"
        )

    def _init_shared_outputs(self, layout: str):
        collection = self.layout[layout]
        with gr.Tabs():
            collection['outputs'] = {}
            with gr.Tab("Display"):
                collection['outputs']['display'] = gradio_markdown()
            with gr.Tab("LaTeX"):
                collection['outputs']['latex'] = gr.TextArea(label="Result", show_copy_button=True)
            with gr.Tab("txt"):
                collection['outputs']['txt'] = gr.TextArea(label="Result", show_copy_button=True)
            with gr.Tab("formatted"):
                collection['outputs']['formatted'] = gr.TextArea(label="Result", show_copy_button=True)

    def _setup_event_handlers(self):
        """Setup event handlers for all new components"""
        # Permutation radio toggle event - horizontal layout
        for layout_type in self.layout:
            layout = self.layout[layout_type]

            input_box = layout["input"]
            gen_input = layout["generators_input"]
            input_box.submit(fn = self.set_poly,
                inputs = [input_box, gen_input, layout["perm_group"]["input"]],
                outputs = [self.image, self.coefficient_triangle],
                show_api=False
            )

            compute_btn = layout["compute_btn"]
            compute_btn.click(fn = partial(self.solve, layout_type=layout_type),
                inputs = [input_box, gen_input, layout["perm_group"]["input"],
                        layout["constraints"]["df"], layout["methods_btn"]],
                outputs = list(layout['outputs'].values()) + \
                [self.image, self.coefficient_triangle], show_api=False)

            perm_group = layout["perm_group"]
            perm_group["radio"].change(
                fn=self._toggle_perm_group_input,
                inputs=[perm_group["radio"], gen_input],
                outputs=[perm_group["input"]],
                show_api=False
            )

            positive_toggle = layout["constraints"]["positive_toggle"]
            for event in (gen_input.blur, gen_input.submit):
                # gen_input.change will update too frequently, so we use blur/submit instead
                event(
                    fn=self._configure_generators,
                    inputs=[gen_input, perm_group["radio"], positive_toggle],
                    outputs=[gen_input, perm_group["input"], layout["constraints"]["df"]],
                    show_api=False
                )


            tmp = layout["constraints"]
            positive_toggle = tmp["positive_toggle"]

            def add_constraint_clicked(df: list, prefix="ineq"):
                df = df[:]
                tp = "≥0" if prefix == "ineq" else "=0"
                df.append(["", "", tp])
                return [df, gr.update(value="Positive", variant="secondary")]

            for prefix in ('ineq', 'eq'):
                add_btn = tmp[f"add_{prefix}uality"]
                add_btn.click(
                    fn=partial(add_constraint_clicked, prefix=prefix),
                    inputs=tmp["df"],
                    outputs=[tmp["df"], positive_toggle],
                    show_api=False
                )

            clear_btn = tmp["clear"]
            clear_btn.click(
                fn=lambda: [[], gr.update(value="Positive", variant="secondary")],
                inputs=[],
                outputs=[tmp["df"], positive_toggle],
                show_api=False
            )

            positive_toggle.click(
                fn=self._toggle_positive,
                inputs=[positive_toggle, tmp["df"], layout["generators_input"]],
                outputs=[positive_toggle, tmp["df"]],
                show_api=False
            )

    def _setup_api(self):
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

    def _render_heatmap(self, grid):
        if grid is None:
            return None
        image = grid.save_heatmap(backgroundcolor=255)
        image = np.vstack([np.full((8, image.shape[1], 3), 255, np.uint8), image])
        side_white = np.full((image.shape[0], 4, 3), 255, np.uint8)
        image = np.hstack([side_white, image, side_white])
        image = Image.fromarray(image).resize((300,300), Image.LANCZOS)
        return image

    def _render_coefficient_triangle(self, degree: int, expr: Poly) -> str:
        """
        Render the coefficient triangle HTML for the given polynomial.

        Args:
            degree (int): The degree of the polynomial.
            expr (Expr): The polynomial expression.

        Returns:
            str: The HTML string for the coefficient triangle.
        """
        if expr is None or not isinstance(expr, Poly):
            return None
        html = '<div id="coeffs" style="width: 100%; height: 600px; position: absolute;">'

        n = degree
        coeffs = expr.coeffs()
        monoms = expr.monoms()
        monoms.append((-1,-1,0))  # tail flag

        l = 100. / (1 + n)
        ulx = (50 - l * n / 2)
        uly = (29 - 13 * n * l / 45)
        fontsize = max(11, int(28-1.5*n))
        lengthscale = .28 * fontsize / 20.
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

    def _toggle_perm_group_input(self, radio_value: str, gen_input: str):
        """Toggle permutation input box editable state based on radio selection"""
        updates = {"interactive": radio_value == "Custom"}
        n = len(gen_input)
        if radio_value == "Cyc" or radio_value == "Sym":
            value = [list(range(1, n)) + [0]] if n > 0 else [[]]
            if radio_value == "Sym" and n > 2:
                value.append([1, 0] + list(range(2, n)))
            updates["value"] = value
        return gr.update(**updates)

    def _configure_generators(self, input_text, perm_group_radio, positive_toggle):
        """Configure generators input"""
        # only lowercase a-z, no duplicates
        filtered = re.sub(r'[^a-z]', '', input_text)
        # Remove duplicates
        unique_chars = []
        for char in filtered:
            if char not in unique_chars:
                unique_chars.append(char)
        chars = ''.join(unique_chars)

        if perm_group_radio == "Cyc" or perm_group_radio == "Sym":
            n = len(chars)
            perm_group_output = [list(range(1, n)) + [0]] if n > 0 else [[]]
            if perm_group_radio == "Sym" and n > 2:
                perm_group_output.append([1, 0] + list(range(2, n)))
            perm_group_output = gr.update(value=perm_group_output)
        else:
            perm_group_output = gr.update()

        if LOCK_MARK in positive_toggle:
            constraints = gr.update(value=[[gen, "", "≥0"] for gen in chars])
        else:
            constraints = gr.update()

        return chars, perm_group_output, constraints

    def _toggle_positive(self, button_value, constraints, gens):
        """Handle positive toggle button functionality"""
        # Toggle button state
        lock = LOCK_MARK

        locked = False
        if not (lock in button_value):
            locked = True

        btn = gr.update(
            value=f"Positive {lock}" if locked else "Positive",
            variant="primary" if locked else "secondary",
        )
        if locked:
            constraints = [[gen, "", "≥0"] for gen in gens]
        return btn, constraints

    def set_poly(self,
        input_text: str,
        gens: str,
        perm_group: str,
    ):
        if isinstance(gens, str):
            gens = tuple([Symbol(g) for g in gens])
        if len(gens) != 3:
            # do not visualize
            return {self.image: None, self.coefficient_triangle: None}

        try:
            perm_group = SOSManager.parse_perm_group(perm_group)
            parser = SOSManager.make_parser(gens, perm_group)

            expr = parser(input_text, return_type="expr",
                    parse_expr_kwargs = {"evaluate": False})
            if expr is None:
                return {self.image: None, self.coefficient_triangle: None}
        except (ValueError, TypeError, SyntaxError, TokenError) as e:
            raise gr.Error(f"{e.__class__.__name__}: {e}")

        frac = parser(expr, return_type="frac")
        if frac[0] is None:
            return {self.image: None, self.coefficient_triangle: None}

        degree, triangle, heatmap = SOSManager.render_coeff_triangle_and_heatmap(
            expr, frac, return_grid=True
        )

        return {
            self.image: self._render_heatmap(heatmap),
            self.coefficient_triangle: \
                self._render_coefficient_triangle(degree, frac[0])
        }


    def solve(self,
        input_text,
        gens,
        perm_group,
        constraints,
        methods,
        layout_type = None
    ):
        """Common solving logic for both horizontal and vertical layouts.

        Args:
            input_text: Input polynomial string
            gens: List of generators
            perm_group: Permutation group
            constraints: List of constraints
            methods: List of methods to use for solving
            layout_type: Layout type ('horizontal' or 'vertical') if specific layout output is needed

        Returns:
            Dictionary with results for the specified layout, or tuple for compatibility with original functions
        """
        # Get polynomial and grid from set_poly
        if isinstance(gens, str):
            gens = tuple([Symbol(g) for g in gens])
        try:
            perm_group = SOSManager.parse_perm_group(perm_group)
            parser = SOSManager.make_parser(gens, perm_group)

            expr = parser(input_text, return_type="expr",
                    parse_expr_kwargs = {"evaluate": False})
        except (ValueError, TypeError, SyntaxError, TokenError) as e:
            raise gr.Error(f"{e.__class__.__name__}: {e}")

        solution = None

        render = self.set_poly(input_text, gens, perm_group)

        if expr is not None:
            ineq_constraints = {}
            eq_constraints = {}

            for constraint in constraints:
                if len(constraint) != 3:
                    continue
                if constraint[2] == "≥0":
                    ineq_constraints[constraint[0]] = constraint[1]
                elif constraint[2] == "=0":
                    eq_constraints[constraint[0]] = constraint[1]

            try:
                ineq_constraints = SOSManager.parse_constraints_dict(ineq_constraints, parser)
                eq_constraints = SOSManager.parse_constraints_dict(eq_constraints, parser)
            except (ValueError, TypeError, SyntaxError, TokenError) as e:
                raise gr.Error(f"{e.__class__.__name__}: {e}")

        if expr is not None:
            try:
                # Attempt to find sum of squares
                solution = SOSManager.sum_of_squares(
                    expr,
                    ineq_constraints,
                    eq_constraints,
                    methods=['%sSOS' % method for method in methods] + ['Pivoting', 'Reparametrization'],
                )
            except Exception as e:
                pass


        if solution is not None:
            # Prepare LaTeX output
            lhs_expr = Symbol('\\text{LHS}')

            tex = solution.to_string(mode='latex', lhs_expr=lhs_expr,
                        together=True, cancel=True, settings={'long_frac_ratio': 2})

            if GRADIO_LATEX_SUPPORTS_ALIGNED:
                tex = recursive_latex_auto_linebreak(tex)

            tex = '$$%s$$' % tex
            gradio_latex = _convert_to_gradio_latex(tex)

            # Format solution strings
            solution_txt = solution.to_string(mode='txt', lhs_expr=lhs_expr)
            solution_formatted = solution.to_string(mode='formatted', lhs_expr=lhs_expr)

            return (
                gradio_latex, tex, solution_txt, solution_formatted,
                render[self.image], render[self.coefficient_triangle],
            )
        else:
            # No solution found
            no_solution = "No solution found"
            return (
                no_solution, no_solution, no_solution, no_solution,
                render[self.image], render[self.coefficient_triangle],
            )

    def toggle_layout(self, vertical_mode: bool, *args, **kwargs):
        """Toggle between horizontal and vertical layout"""
        # Toggle the mode
        new_mode = not vertical_mode

        # Sync values between layouts

        updates = [None] * len(args)
        for i in range(0, len(args), 2):
            if new_mode:
                updates[i] = gr.update()
                updates[i+1] = args[i]
            else:
                updates[i] = args[i+1]
                updates[i+1] = gr.update()

        # Return update objects for each output component in the order expected by the outputs list
        return [
            new_mode,  # Update vertical_mode state
            gr.update(visible=not new_mode),  # Update horizontal container visibility
            gr.update(visible=new_mode),  # Update vertical container visibility
        ] + updates

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
                parsed_expr,
                parsed_ineqs,
                parsed_eqs
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
    SOSManager.verbose = False
    SOSManager.time_limit = 300.0

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
                # "http://localhost:5173",             # Vite
                # "http://127.0.0.1:5173"              # local
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"]
        )

    interface.demo.launch(show_error=True) #, debug=True)
