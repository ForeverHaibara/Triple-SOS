import numpy as np

from ...utils import generate_monoms

def get_features(problem):
    """
    Experimental features of a problem. This is used for learning-based algorithms.
    The features might vary across versions.

    Temporarily supports polynomial problems only.
    """
    from scipy.stats import skew
    _skew = lambda x:skew(x) if np.max(x) != np.min(x) else 0
    _mean = lambda x, default=0: np.mean(x) if len(x) else default
    _divide = lambda x, y: x/y if y != 0 else 0

    ineq_constraints, eq_constraints = problem.ineq_constraints, problem.eq_constraints

    homogeneous = problem.is_homogeneous
    degree_list = problem.expr.degree_list()

    QUANTILES = 5

    coeffs          = problem.expr.coeffs()
    coeffs          = np.array(coeffs).astype(float)
    abscoeffs       = np.abs(coeffs)
    symmetry        = problem.identify_symmetry()
    coeffs_quant    = np.quantile(coeffs, np.linspace(0, 1, QUANTILES))
    abscoeffs_quant = np.quantile(abscoeffs, np.linspace(0, 1, QUANTILES))
    nvars           = len(problem.free_symbols)
    dt = {
        'nvars':           nvars,
        'terms':           len(coeffs),
        'sparsity':        len(coeffs)/(max(1, len(generate_monoms(nvars, problem.expr.total_degree(), hom=homogeneous)[1]))),
        'degree':          problem.reduce(lambda x:x.total_degree(), max),
        'ineqs_num':       len(ineq_constraints),
        'ineqs_deg_max':   np.max([c.total_degree() for c in ineq_constraints]) if ineq_constraints else 0,
        'ineqs_deg_mean':  _mean([c.total_degree() for c in ineq_constraints]),
        'ineqs_linear':    _mean([c.is_linear for c in ineq_constraints], 1),
        'ineqs_binom':     _mean([len(c.coeffs()) <= 2 for c in ineq_constraints], 1),
        'ineqs_terms':     np.sum([len(c.coeffs()) for c in ineq_constraints]),
        'eqs_num':         len(eq_constraints),
        'eqs_deg_max':     np.max([c.total_degree() for c in eq_constraints]) if eq_constraints else 0,
        'eqs_deg_mean':    _mean([c.total_degree() for c in eq_constraints]),
        'eqs_linear':      _mean([c.is_linear for c in eq_constraints], 1),
        'eqs_binom':       _mean([len(c.coeffs()) <= 2 for c in eq_constraints], 1),
        'eqs_terms':       np.sum([len(c.coeffs()) for c in eq_constraints]),
        'coeffs_mean':     _mean(coeffs),
        'coeffs_var':      coeffs.var(),
        'coeffs_ng_mean':  _mean(coeffs < 0),
        'coeffs_skew':     _skew(coeffs),
        'coeffs_quant':    tuple(coeffs_quant.tolist()),
        'coeffs_out':      (coeffs > coeffs_quant[3] + 1.5*(coeffs_quant[3]-coeffs_quant[1])).sum(),
        'abscoeffs_mean':  abscoeffs.mean(),
        'abscoeffs_var':   abscoeffs.var(),
        'abscoeffs_skew':  _skew(abscoeffs),
        'abscoeffs_quant': tuple(abscoeffs_quant.tolist()),
        'abscoeffs_out':   (abscoeffs > abscoeffs_quant[3] + 1.5*(abscoeffs_quant[3]-abscoeffs_quant[1])).sum(),
        'degree_list':     degree_list,
        'homogeneous':     homogeneous,
        'rational':        problem.reduce(lambda x:x.domain.is_QQ or x.domain.is_ZZ),
        'symmetry_order':  symmetry.order(),
        'symmetry_sym':    symmetry.is_symmetric,
        'symmetry_cyc':    symmetry.is_cyclic
    }

    dt['coeffs_mean']  = _divide(dt['coeffs_mean'], dt['abscoeffs_mean'])
    dt['coeffs_cv']    = _divide(dt['coeffs_var']**0.5, dt['abscoeffs_mean'])
    dt['abscoeffs_cv'] = _divide(dt['abscoeffs_var']**0.5, dt['abscoeffs_mean'])
    for i in range(5):
        dt[f'coeffs_quant_{i}']    = _divide(dt['coeffs_quant'][i], dt['abscoeffs_mean'])
        dt[f'abscoeffs_quant_{i}'] = _divide(dt['abscoeffs_quant'][i], dt['abscoeffs_mean'])

    for key in dt:
        if isinstance(dt[key], (np.floating, np.integer)):
            dt[key] = float(dt[key])
        if isinstance(dt[key], (np.ndarray)):
            dt[key] = dt[key].tolist()
        if isinstance(dt[key], list):
            dt[key] = tuple(dt[key])
    return dt
