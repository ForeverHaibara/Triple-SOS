def test_dispatch_names():
    # import all singledispatch functions from ..dispatch
    import importlib
    dispatch = importlib.import_module('..dispatch', __package__)

    funcs = [getattr(dispatch, f) for f in dir(dispatch) if f.startswith('_dtype_')]
    funcs = [f for f in funcs if callable(f) and hasattr(f, 'registry')]

    for f in funcs:
        name = f.__name__[7:] # strip "_dtype_"
        for g in f.registry.values():
            assert name in g.__name__,\
                f"{g.__name__} does not match the name of {f.__name__} as a singledispatch registry"

    # open the dispatch.py to check no data type is registered twice
    # TODO: use a more robust way to check the registry?
    import os
    dispatch_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../dispatch.py')
    with open(dispatch_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    registry = set()
    for line in lines:
        if line.startswith('@_dtype_') and '.register(' in line:
            line = line.rstrip()
            if line in registry:
                assert False, f"{line} is registered twice in {dispatch_path}"
            registry.add(line)