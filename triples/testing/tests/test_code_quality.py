def _project_path():
    from os.path import dirname, abspath
    return dirname(dirname(dirname(abspath(__file__))))


def test_ruff():
    """
    Execute ruff on the project.
    """
    import subprocess

    from platform import python_version_tuple

    # Get the project root directory
    project_root = _project_path()

    try:
        if int(python_version_tuple()[0]) >= 3 and\
           int(python_version_tuple()[1]) >= 7:
            kwargs = {"capture_output": True, "text": True}
        else:
            kwargs = {}

        # Run ruff on the project
        result = subprocess.run(
            ["ruff", "check", project_root],
            **kwargs
        )
    except FileNotFoundError:
        # pytest.skip("Ruff is not installed in the environment")
        return

    # Assert that ruff passes without errors
    assert result.returncode == 0, (
        f"Ruff found issues:\n{result.stdout}\n{result.stderr}"
    )


def test_eof_newline():
    # ruff does not check W391 in the stable version
    # so we use the manual check
    from os import walk
    from os.path import join, relpath
    lib_path = _project_path()
    for root, dirs, files in walk(lib_path):
        for file in files:
            if file.endswith(".py"):
                file_path = join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.read()
                except Exception:
                    # print(f"Error reading {file_path}")
                    continue
                assert not lines.endswith("\n\n"),\
                    f"EOF multiple newlines found in {join('triples', relpath(file_path, lib_path))}"
