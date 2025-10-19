def test_dependency():
    """
    Test to ensure all import statements in .py files under a target directory only use allowed libraries.
    Allowed libraries: sympy, numpy, scipy, Python standard libraries, and the library's own modules.
    """
    import sys
    import os

    # path to the parent directory
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                    if not (imp in ALLOWED_LIBS):
                        if imp == "pytest" and file.startswith("test_"):
                            continue
                        forbidden_libraries.add((file_path, line_num, imp))

    if forbidden_libraries:
        forbidden_libraries = sorted(list(forbidden_libraries))
        message = '\n'.join([f"{fp}:line {ln}:import {lib}"%(fp, ln, lib) for fp, ln, lib in forbidden_libraries])
        assert False, (
            f"Forbidden dependencies detected: {message}."
        )
