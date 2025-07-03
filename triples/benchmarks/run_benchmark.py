BENCHMARKS = []

def _append_paths(paths=None):
    if paths:
        if isinstance(paths, str):
            paths = [paths]
        import sys
        [sys.path.append(path) for path in paths]

def _get_git_commit_id(digits=None):
    import subprocess
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        if digits is not None:
            commit_id = commit_id[:digits]
    except:
        commit_id = ""
    return commit_id

def _get_platform_info(verbose=True):
    import platform
    PYTHON_VERSION = platform.python_version()
    from sympy import __version__ as SYMPY_VERSION
    from numpy import __version__ as NUMPY_VERSION
    from scipy import __version__ as SCIPY_VERSION

    GMPY2_VERSION = ""
    try:
        from gmpy2 import __version__ as GMPY2_VERSION
    except ImportError:
        pass

    FLINT_VERSION = ""
    try:
        from flint import __version__ as FLINT_VERSION
    except ImportError:
        pass

    CPU = platform.processor()
    OS = platform.platform()

    try:
        import psutil
        RAM = psutil.virtual_memory().total / 1024**3
    except ImportError:
        RAM = None

    platform_info = {
        'os': OS,
        'python': PYTHON_VERSION,
        'sympy': SYMPY_VERSION,
        'numpy': NUMPY_VERSION,
        'scipy': SCIPY_VERSION,
        'gmpy2': GMPY2_VERSION,
        'flint': FLINT_VERSION,
        'cpu': CPU,
        'ram': RAM,
    }
    if verbose:
        print('='*80)
        print('Platform Information')
        print('='*80)
        for key, value in platform_info.items():
            print(f'{key.ljust(10)}: {value}')
        print('='*80)

    return platform_info

def _save_batch(result, f, cnt):
    import json
    # pad cnt with at least 4 digits
    cnt = str(cnt).zfill(4)
    with open(f"{f}/{cnt}.jsonl", 'w') as f:
        for item in result:
            json.dump(item, f)
            f.write('\n')
    # print(f"Save {cnt} batch to {f}/{cnt}.jsonl")



def run_bench(benchmarks=BENCHMARKS, save_interval=600):
    platform_info = _get_platform_info()

    from triples.core import sum_of_squares
    from datetime import datetime
    from tempfile import TemporaryDirectory
    import json

    DATETIME_LEN = 15
    start_bench_time = datetime.now()
    last_save_time = start_bench_time
    save_cnt = 0
    git_commit_id = _get_git_commit_id()
    bench_id = f"{start_bench_time.strftime('%Y%m%d-%H%M%S')}{('-'+git_commit_id[:7]) if git_commit_id else ''}"

    results = []
    batch_results = []

    with TemporaryDirectory(prefix=f".bench-{start_bench_time.strftime('%Y%m%d-%H%M%S')}", delete=False) as temp_dir:
        header = {
            "id": bench_id,
            "commit_hash": git_commit_id,
            "timestamp": start_bench_time.isoformat(),
            "platform": platform_info,
        }
        json.dump(header, open(f"{temp_dir}/{bench_id}-header.json", 'w'), indent=4)
        print(f"Tempfile saved to {temp_dir}/{bench_id}-header.json")

        for problem_set in benchmarks:
            problem_set_name = problem_set.__name__
            problems = problem_set.collect(exclude=mark.noimpl)
            instance = problem_set()

            total = len(problems)
            skipped, success, failed = 0, 0, 0

            max_problem_name_len = max([len(problem_name) for problem_name in problems.keys()], default=0)
            for i, (problem_name, problem) in enumerate(problems.items(), start=1):
                # (problem_name) (status) (start_time) --- (end_time) [(solved)/(processed)]
                problem_marks = getattr(problem, 'marks', tuple())
                print(problem_name.ljust(max_problem_name_len), end=' ')
                if mark.skip in problem_marks:
                    skipped += 1
                    batch_results.append(
                        {'set': problem_set_name, 'name': problem_name, 'status': 'skipped',
                        'time': None, 'solution': None, 'length': None}
                    )
                    print(f"s{(7+2*DATETIME_LEN)*' '}[{success/total*100:.2f}%/{i/total*100:.2f}%]")
                else:
                    start_time = datetime.now()
                    print(start_time.time(), end='')
                    solution = None
                    try:
                        solution = sum_of_squares(*problem(instance))
                    except Exception as e:
                        if isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise e
                        solution = None
                    end_time = datetime.now()
                    elapsed_time = (end_time-start_time).total_seconds()
                    if solution is not None:
                        success += 1
                        status_code = '.'
                        sol_string = str(solution.solution)
                        batch_results.append(
                            {'set': problem_set_name, 'name': problem_name, 'status': 'success',
                            'time': elapsed_time, 'solution': sol_string, 'length': len(sol_string)}
                        )
                    else:
                        failed += 1
                        status_code = '\033[91mF\033[0m'
                        batch_results.append(
                            {'set': problem_set_name, 'name': problem_name, 'status': 'failed',
                            'time': elapsed_time, 'solution': None, 'length': None}
                        )
                    print(f"{'\b'*DATETIME_LEN}{status_code} {start_time.time()} --- {end_time.time()} "\
                            + f"[{success/total*100:.2f}%/{i/total*100:.2f}%]")

                    if (end_time - last_save_time).total_seconds() >= save_interval:
                        _save_batch(batch_results, temp_dir, save_cnt)
                        save_cnt += 1
                        last_save_time = end_time
                        results.extend(batch_results)
                        batch_results = []

        _save_batch(batch_results, temp_dir, save_cnt)
        results.extend(batch_results)
        batch_results = []

    header["results"] = results
    import os
    os.makedirs("./.benchmarks", exist_ok=True)
    with open(f"./.benchmarks/bench-{bench_id}.json", 'w') as f:
        json.dump(header, f, indent=4)
    print(f"Benchmark results saved to ./.benchmarks/bench-{bench_id}.json")

    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    _append_paths(paths=['./', '../../'])

    from problems import ProblemSet, mark
    from problems import (
        IMOProblems, IMOSLProblems
    )

    BENCHMARKS = [
        # IMOProblems,
        IMOSLProblems
    ]

    run_bench(BENCHMARKS)