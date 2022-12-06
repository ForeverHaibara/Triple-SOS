import numpy as np
import sympy as sp

# from root_guess import verify_isstrict
from .text_process import PreprocessText, cycle_expansion, next_permute, reflect_permute, deg


def arraylize(poly, dict_monom: dict, inv_monom: list):
    '''
    Turn a sympy polynomial into arraylike representation.
    '''
    coeffs = np.zeros(len(inv_monom))
    for coeff, monom in zip(poly.coeffs(),poly.monoms()):
        i , j , k = monom 
        if i >= j and (i > k or (i == k and i == j)):
            coeffs[dict_monom[(i,j,k)]] = coeff
    return coeffs 


def arraylize_sp(poly, dict_monom: dict, inv_monom: list):
    '''
    Turn a sympy polynomial into sympy-arraylike representation.
    '''
    coeffs = sp.zeros(len(inv_monom), 1)
    for coeff, monom in zip(poly.coeffs(),poly.monoms()):
        i , j , k = monom 
        if i >= j and (i > k or (i == k and i == j)):
            coeffs[dict_monom[(i,j,k)]] = coeff
    return coeffs 


def invarraylize(poly, dict_monom: dict, inv_monom: list):
    '''
    Turn a sympy polynomial into arraylike representation.
    '''
    val = ''
    for monom, coeff in enumerate(poly):
        if coeff == 0:
            continue
        i , j , k = inv_monom[monom]
        val += f'+ {coeff} * a^{i} * b^{j} * c^{k}'
    return sp.polys.polytools.Poly(cycle_expansion(val))
    

def generate_expr(n: int):
    '''
    For cyclic, homogenous, 3-variable polynomials, only part of monoms are needed:

    a^i * b^j * c^k where (i>=j and i>k) or (i==j==k)

    inv_monoms records all such (i,j,k)
    dict_monom records the index of each (i,j,k) in inv_monoms
    '''
    dict_monom = {}
    inv_monom = []
    for i in range((n-1)//3+1, n+1):
        # 0 <= k = n - i - j < i
        for j in range(max(0,n-2*i+1), min(i+1,n-i+1)):
            inv_monom.append((i,j,n-i-j))
            dict_monom[inv_monom[-1]] = len(inv_monom) - 1
    
    if n % 3 == 0:
        # i = j = k = n//3
        inv_monom.append((n//3,n//3,n//3))
        dict_monom[inv_monom[-1]] = len(inv_monom) - 1

    return dict_monom, inv_monom


def base_square(m: int) -> str:
    '''
    (a-b)^2i * (b-c)^2j * (c-a)^2k

    where 2i+2j+2k = 2m  and  k <= i
    '''
    for i in range(0, m+1):
        # 0 <= k = m-i-j <= i
        for j in range(0, m-i+1):
            prefix = f'(a-b)^{2*i} * (b-c)^{2*j} * (c-a)^{2*(m-i-j)}'
            yield prefix
            

def base_trivial(m: int) -> str:
    '''
    a^i * b^j * c^k 
    where i+j+k = m  and  k <= i
    '''
    for i in range((m-1)//3+1, m+1):
        # 0 <= k = m2-i2-j2 <= i2
        for j in range(max(0,m-2*i), min(i+1,m-i+1)):
            name = f'a^{i} * b^{j} * c^{m-i-j}'
            yield name
            
def base_tangent(m: int, tangents: list) -> str:
    '''
    (a-b)^2i * (b-c)^2j * (c-a)^2k * a^u * b^v * c^w * f(a,b,c)^2t
    where its degree is m
    '''
    for tangent in tangents:
        try:
            n = deg(sp.polys.polytools.Poly(tangent))
        except:
            continue
        for _ in range(3):
            for i in range(1, m//(2*n)+1):
                for j in range(0, (m-2*n*i)//2+1):
                    for prefix in base_square(j):
                        for suffix in base_trivial(m-2*(n*i+j)):
                            name = prefix + ' * ' + suffix + ' * (' + tangent + f')^{2*i}'
                            yield name
            tangent = next_permute(tangent)


def append_basis(n: int, dict_monom: dict, inv_monom: list, 
                names: list, polys: list, basis: np.ndarray, tangents = None):
    '''
    Add basis generated from tangents.
    '''
    if not tangents:
        return names, polys, basis

    old_names = len(names)
    for i in range(len(tangents)):
        tangents[i] = PreprocessText(tangents[i],cyc=False,retText=True)

    for mixed_tangent in base_tangent(n, tangents):
        names.append(mixed_tangent)
    
    if old_names == len(names):
        return names, polys, basis

    for name in names[old_names:]:
        polys.append(sp.polys.polytools.Poly(cycle_expansion(name)))
        
    new_basis = np.array([arraylize(poly,dict_monom,inv_monom) for poly in polys[old_names:]])
    if type(basis) == list and len(basis) == 0:
        basis = new_basis
    else:
        basis = np.vstack((basis, new_basis))

    return names, polys, basis


def generate_basis(n: int, dict_monom: dict, inv_monom: list, tangents = None, strict_roots = None, aux = None):
    '''
    Return
    -------
    names: list of str, e.g.
        ['(a+b-c)^2']
    
    polys: list of sympy polynomials, e.g. 
        [sp.polys.polytools.Poly('(a+b-c)^2+(b+c-a)^2+(c+a-b)^2')]

    basis: ndarray
        Vstack of ndarray representation of each polynomials.
    '''
    names = []
    polys = []
    basis = []

    nontrivial_roots = False
    if strict_roots is not None and len(strict_roots) > 0:
        for (a, b) in strict_roots:
            if min((abs(a-1), abs(b-1), abs(a-b), abs(a), abs(b))) > 5e-3:
                nontrivial_roots = True
                break

    if not nontrivial_roots:

        for m in range(0,n//2+1):
            # ((a-b)^i (b-c)^j (c-a)^k)^2    where 2 <= 2(i+j+k) = 2m <= n
            # and    i+j+k = m

            # m2 = n - 2m,    standing for the remaining orders excluding the squares
            m2 = n - 2*m
            for prefix in base_square(m):
                for suffix in base_trivial(m2):
                    names.append(prefix + ' * ' + suffix)

        for i in range(1,n-1):
            for j in range(1, n-i):
                k = n - i - j
                name = f'a^{i+1}*b^{j}*c^{k-1} + a^{i}*b^{j+1}*c^{k-1} + a^{i-1}*b^{j-1}*c^{k+2} - 3*a^{i}*b^{j}*c^{k}'
                names.append(name)
            
        if aux is None:
            aux = []
        if n == 3:
            aux = ['a^2*b-a*b*c','a*b^2-a*b*c']
        elif n == 5:
            aux = []#['a*(a-b)^2*(a+b-c)^2','b*(a-b)^2*(a+b-c)^2','c*(a-b)^2*(a+b-c)^2']
        elif n == 6:
            aux = ['a*b*c*a*(a-b)*(a-c)','a^2*(a^2-b^2)*(a^2-c^2)','a*b*c*(a^2*b-a*b*c)',
                    'a*b*c*(a*b^2-a*b*c)','a*b*(a*b-b*c)*(a*b-c*a)']
        names = names + aux
        names.append(f'a^{n-2}*(a-b)*(a-c)')
        
        polys = [sp.polys.polytools.Poly(cycle_expansion(name)) for name in names]
        basis = np.array([arraylize(poly,dict_monom,inv_monom) for poly in polys])


    names, polys, basis = append_basis(n, dict_monom, inv_monom, names, polys, basis, tangents)

    return names, polys, basis


def reduce_basis(n, dict_monom, inv_monom, names, polys, basis, strict_roots, tol=1e-3):
    '''delete the basis that are not zero in strict roots'''

    if strict_roots is None or len(strict_roots) == 0 or type(basis) == list:
        return names, polys, basis
    
    m = basis.shape[0]
    mask = [1] * m

    for (a,b) in strict_roots:
        vals = []
        powera = [a**i for i in range(n+1)]
        powerb = [b**i for i in range(n+1)]
        for (i,j,k) in inv_monom:
            if i != k:
                vals.append(powera[i]*powerb[j]+powera[j]*powerb[k]+powera[k]*powerb[i])
            else: # i==j==k
                vals.append(powera[i]*powerb[j])
        vals = np.array(vals)

        vals = basis @ vals.T 
        for i in range(m):
            if abs(vals[i]) > tol: # nonzero
                mask[i] = 0
    
    mask = [i for i in range(m) if mask[i]]
    names = [names[i] for i in mask]
    polys = [polys[i] for i in mask]
    basis = basis[mask]

    return names, polys, basis
    