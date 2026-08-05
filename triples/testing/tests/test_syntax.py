"""Static checks for syntax which is incompatible with supported dependencies."""


import ast
import os


# Add compatibility restrictions here.
COMPATIBILITY_RULES = {
    "imports": (
        {
            "module": "sympy.external.gmpy",
            "name": "lcm",
            "message": "lcm is not available in SymPy 1.9",
        },
        {
            "module": "sympy.matrices.matrixbase",
            "name": "MatrixBase",
            "message": "sympy.matrices.matrixbase is not available in SymPy 1.9",
        }
    ),
    "attributes": (
        {
            "owner": "sympy.polys.matrices.DomainMatrix",
            "name": "from_list",
            "message": "from_list is not available in SymPy 1.9",
        },
    ),
}


def _project_path():
    from os.path import dirname, abspath
    return dirname(dirname(dirname(abspath(__file__))))


def _source_files(base_path):
    """Yield package source files which use the supported dependencies."""
    modules = ("utils", "core", "sdp")
    for module in modules:
        for root, _, files in os.walk(os.path.join(base_path, module)):
            for file in files:
                if file.endswith(".py"):
                    yield os.path.join(root, file)


def _node_name(node):
    """Return a dotted name for a simple AST name or attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _node_name(node.value)
        if parent is not None:
            return parent + "." + node.attr
    return None


def _resolve_name(node, imports):
    """Resolve an imported alias to its fully-qualified name."""
    name = _node_name(node)
    if name is None:
        return None

    parts = name.split(".")
    imported_name = imports.get(parts[0])
    if imported_name is None:
        return name if name.startswith("sympy.") else None
    if len(parts) == 1:
        return imported_name
    return imported_name + "." + ".".join(parts[1:])


def _target_name(node):
    """Return a stable name for a simple assignment target."""
    if isinstance(node, ast.arg):
        return node.arg
    return _node_name(node)


class _ImportCollector(ast.NodeVisitor):
    def __init__(self):
        self.imports = {}

    def visit_Import(self, node):
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            imported_name = alias.name if alias.asname else alias.name.split(".")[0]
            self.imports[local_name] = imported_name

    def visit_ImportFrom(self, node):
        if node.module is None:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            self.imports[local_name] = node.module + "." + alias.name


class _TypeCollector(ast.NodeVisitor):
    """Collect types which can be inferred without executing project code."""

    def __init__(self, imports):
        self.imports = imports
        self.types = {}

    def _annotation_type(self, node):
        annotation = _resolve_name(node, self.imports)
        if annotation is not None and annotation.startswith("sympy."):
            return annotation
        return None

    def _value_type(self, node):
        if isinstance(node, ast.Call):
            value_type = _resolve_name(node.func, self.imports)
            if value_type is not None and value_type.startswith("sympy."):
                return value_type
        value_name = _target_name(node)
        if value_name is not None:
            return self.types.get(value_name)
        return None

    def _save_target_type(self, target, value_type):
        target_name = _target_name(target)
        if target_name is not None and value_type is not None:
            self.types[target_name] = value_type

    def visit_arg(self, node):
        value_type = self._annotation_type(node.annotation)
        self._save_target_type(node, value_type)

    def visit_AnnAssign(self, node):
        value_type = self._annotation_type(node.annotation)
        if value_type is None and node.value is not None:
            value_type = self._value_type(node.value)
        self._save_target_type(node.target, value_type)
        self.generic_visit(node)

    def visit_Assign(self, node):
        value_type = self._value_type(node.value)
        for target in node.targets:
            self._save_target_type(target, value_type)
        self.generic_visit(node)


class _CompatibilityChecker(ast.NodeVisitor):
    def __init__(self, imports, types, rules, file_path):
        self.imports = imports
        self.types = types
        self.rules = rules
        self.file_path = file_path
        self.violations = []

    def _add_violation(self, node, rule, description):
        message = rule.get("message", description)
        self.violations.append(
            (self.file_path, node.lineno, node.col_offset + 1, message)
        )

    def visit_ImportFrom(self, node):
        module = node.module
        for alias in node.names:
            for rule in self.rules.get("imports", ()):
                if module == rule["module"] and alias.name == rule["name"]:
                    self._add_violation(
                        node,
                        rule,
                        "forbidden import {}.{}".format(module, alias.name),
                    )
        self.generic_visit(node)

    def _attribute_owner(self, node):
        value_type = _resolve_name(node.value, self.imports)
        if value_type is not None and value_type.startswith("sympy."):
            return value_type

        value_name = _target_name(node.value)
        return self.types.get(value_name)

    def visit_Attribute(self, node):
        owner = self._attribute_owner(node)
        if owner is not None:
            for rule in self.rules.get("attributes", ()):
                if owner == rule["owner"] and node.attr == rule["name"]:
                    self._add_violation(
                        node,
                        rule,
                        "forbidden attribute {}.{}".format(owner, node.attr),
                    )
        self.generic_visit(node)


def _find_compatibility_violations(source, file_path="<string>", rules=None):
    """Find configured SymPy compatibility violations in Python source."""
    if rules is None:
        rules = COMPATIBILITY_RULES

    tree = ast.parse(source, filename=file_path)
    import_collector = _ImportCollector()
    import_collector.visit(tree)

    type_collector = _TypeCollector(import_collector.imports)
    type_collector.visit(tree)

    checker = _CompatibilityChecker(
        import_collector.imports,
        type_collector.types,
        rules,
        file_path,
    )
    checker.visit(tree)
    return checker.violations


def test_sympy_compatibility():
    """Check source files against the manually maintained compatibility rules."""
    violations = []
    for file_path in _source_files(_project_path()):
        with open(file_path, "r", encoding="utf-8") as source_file:
            source = source_file.read()
        violations.extend(_find_compatibility_violations(source, file_path))

    assert not violations, "\n".join(
        "{}:{}:{}: {}".format(file_path, line, column, message)
        for file_path, line, column, message in violations
    )


def test_sympy_compatibility_checker():
    """Check imports, class methods, and methods used through instance fields."""
    source = """
from sympy.external.gmpy import forbidden_import
from sympy.external import gmpy
from sympy.polys.matrices import DomainMatrix as MatrixDomain

class SomeClass:
    def __init__(self, x: MatrixDomain):
        self.x = x

    def method(self):
        self.x.some_func()
        MatrixDomain.some_other_func()
        gmpy.lcm(1, 2)
"""
    rules = {
        "imports": (
            {
                "module": "sympy.external.gmpy",
                "name": "forbidden_import",
                "message": "forbidden import is unavailable",
            },
        ),
        "attributes": (
            {
                "owner": "sympy.external.gmpy",
                "name": "lcm",
                "message": "gmpy.lcm is unavailable",
            },
            {
                "owner": "sympy.polys.matrices.DomainMatrix",
                "name": "some_func",
                "message": "DomainMatrix.some_func is unavailable",
            },
            {
                "owner": "sympy.polys.matrices.DomainMatrix",
                "name": "some_other_func",
                "message": "DomainMatrix.some_other_func is unavailable",
            },
        ),
    }

    violations = _find_compatibility_violations(source, rules=rules)
    assert [message for _, _, _, message in violations] == [
        "forbidden import is unavailable",
        "DomainMatrix.some_func is unavailable",
        "DomainMatrix.some_other_func is unavailable",
        "gmpy.lcm is unavailable",
    ]


def test_dependency():
    """
    Test to ensure all import statements in .py files under a target directory only use allowed libraries.
    Allowed libraries: sympy, numpy, scipy, Python standard libraries, and the library's own modules.
    """
    import sys
    import os

    # path to the parent directory
    base_path = _project_path()

    # modules to be checked
    modules = ["utils", "core", "sdp"]

    # Allowed external libraries
    THIRD_PARTY = {"sympy", "numpy", "scipy", "mpmath"}


    # Get standard library module names (Python 3.10+)
    if not hasattr(sys, 'stdlib_module_names'):
        return
    STDLIB = sys.stdlib_module_names
    ALLOWED_LIBS = set(STDLIB) | THIRD_PARTY

    forbidden_libraries = set()

    # Walk through all .py files in the target directory
    for module in modules:
        for root, _, files in os.walk(os.path.join(base_path, module)):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    # Check for import statements
                    if (not line.startswith("from ")) and (not line.startswith("import ")):
                        continue

                    imp = None
                    if line.startswith("import "):
                        if ',' in line:
                            assert False, "Multiple imports using import statement are not allowed"
                        else:
                            imp = line[len("import "):].strip().split(' ')[0]
                    if line.startswith("from "):
                        imp = line[len("from "):].strip().split(' ')[0]

                    imp = imp.split('.')[0]
                    if len(imp) == 0:
                        # relative import from current package
                        continue
                    if imp not in ALLOWED_LIBS:
                        if imp == "pytest" and file.startswith("test_"):
                            continue
                        forbidden_libraries.add((file_path, line_num, imp))

    if forbidden_libraries:
        forbidden_libraries = sorted(forbidden_libraries)
        message = '\n'.join([f"{fp}:line {ln}:import {lib}"%(fp, ln, lib) for fp, ln, lib in forbidden_libraries])
        assert False, (
            f"Forbidden dependencies detected: {message}."
        )
