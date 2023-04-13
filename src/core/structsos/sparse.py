from math import gcd

def _sos_struct_sparse(poly, degree, coeff, recurrsion, coeffs = None):
    monoms = poly.monoms()
    if len(coeffs) == 1:
        # e.g.  abc
        if coeff(monoms[0]) >= 0:
            monom = monoms[0]
            return [], [coeff(monom) / 3], [f'a^{monom[0]}*b^{monom[1]}*c^{monom[2]}']

    elif len(coeffs) == 3:
        # e.g. (a2b + b2c + c2a)
        if coeff(monoms[0]) >= 0:
            monom = monoms[0]
            return [], [coeff(monom)], [f'a^{monom[0]}*b^{monom[1]}*c^{monom[2]}']

    elif len(coeffs) == 4:
        # e.g. (a2b + b2c + c2a - 8/3abc)
        if coeff(monoms[0]) >= 0 and coeff(monoms[0])*3 + coeff((degree//3, degree//3, degree//3)) >= 0:
            monom = monoms[0]
            return [], \
                [coeff(monom), coeff(monoms[0]) + coeff((degree//3, degree//3, degree//3))/3], \
                [f'a^{monom[0]}*b^{monom[1]}*c^{monom[2]}-a^{degree//3}*b^{degree//3}*c^{degree//3}',
                    f'a^{degree//3}*b^{degree//3}*c^{degree//3}']

    elif len(coeffs) == 6:
        # e.g. s(a5b4 - a4b4c)
        monoms = [i for i in monoms if (i[0]>i[1] and i[0]>i[2]) or (i[0]==i[1] and i[0]>i[2])]
        monoms = sorted(monoms)
        small , large = monoms[0], monoms[1]
        if coeff(small) >= 0 and coeff(large) >= 0:
            return [], \
                [coeff(small), coeff(large)], \
                [f'a^{small[0]}*b^{small[1]}*c^{small[2]}', f'a^{large[0]}*b^{large[1]}*c^{large[2]}']
        elif coeff(large) >= 0 and coeff(large) + coeff(small) >= 0:
            det = 3*large[0]*large[1]*large[2] - (large[0]**3+large[1]**3+large[2]**3)
            deta = small[0]*(large[1]*large[2]-large[0]**2)+small[1]*(large[2]*large[0]-large[1]**2)+small[2]*(large[0]*large[1]-large[2]**2)
            detb = small[0]*(large[2]*large[0]-large[1]**2)+small[1]*(large[0]*large[1]-large[2]**2)+small[2]*(large[1]*large[2]-large[0]**2)
            detc = small[0]*(large[0]*large[1]-large[2]**2)+small[1]*(large[1]*large[2]-large[0]**2)+small[2]*(large[2]*large[0]-large[1]**2)
            det, deta, detb, detc = -det, -deta, -detb, -detc
            # print(det, deta, detb, detc)
            if det > 0 and deta >= 0 and detb >= 0 and detc >= 0:
                d = gcd(det, gcd(deta, gcd(detb, detc)))
                det, deta, detb, detc = det//d, deta//d, detb//d, detc//d
                return [], \
                    [coeff(large)/det, coeff(large) + coeff(small)], \
                    [f'{deta}*a^{large[0]}*b^{large[1]}*c^{large[2]}+{detb}*a^{large[1]}*b^{large[2]}*c^{large[0]}+{detc}*a^{large[2]}*b^{large[0]}*c^{large[1]}-{det}*a^{small[0]}*b^{small[1]}*c^{small[2]}',
                    f'a^{small[0]}*b^{small[1]}*c^{small[2]}']

    return [], None, None