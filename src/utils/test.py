from .text_process import * 

def DegreeofZero(poly):
    '''Compute the degree of a homogeneous zero polynomial
    idea: delete the additions and substractions, which do not affect the degree'''
    poly = poly.lower()
    poly = PreprocessText_DeLatex(poly)
    poly = PreprocessText_Expansion(poly)
    poly = PreprocessText_Completion(poly)
    
    i = 0
    length = len(poly)
    bracket = 0
    while i < length:
        if poly[i] == '+' or poly[i] == '-': 
            # run to the end of this bracket (sum does not affect the degree)
            # e.g. a*(a^2+a*b)*c -> a*(a^2)*c
            bracket_cur = bracket
            j = i + 1
            is_constant = True 
            while j < length:
                if poly[j] == '(':
                    bracket += 1
                elif poly[j] == ')':
                    bracket -= 1
                    if bracket < bracket_cur:
                        break 
                elif poly[j] in 'abc':
                    is_constant = False 
                j += 1
            if is_constant == False:
                poly = poly[:i] + poly[j:]
                length = len(poly)
        elif poly[i] == ')':
            bracket -= 1
        elif poly[i] == '(':
            bracket += 1
            # e.g. a*(-b*c) ,    a*(--+-b*c)
            i += 1
            while i < length and (poly[i] == '-' or poly[i] == '+'):
                i += 1
            if i == length:
                break 
            
        i += 1
        
    try:
    #     degree = deg(sp.polys.polytools.Poly(poly))
        poly = sp.fraction(sp.sympify(poly))
        if poly[1].is_constant():
            degree = deg(sp.polys.polytools.Poly(poly[0]))
        else:
            degree = deg(sp.polys.polytools.Poly(poly[0])) - deg(sp.polys.polytools.Poly(poly[1]))
    except:
        degree = 0
        
    return degree

# print(DegreeofZero('(b3+c3+a3)/s(a)'))
# assert False 
with open(r'D:\Python Projects\Trials\Inequalities\Triples\problems.txt','r') as f:
    data = f.readlines()

from tqdm import tqdm 
for i in tqdm(data):
    i = i[:-1]
    if DegreeofZero(i) != deg(PreprocessText(i)):
        print(i)

    