import numpy as np
from functools import reduce
ean13_decmap = {
    (0,0,0,1,1,0,1):{"digit": 0, "enc": 'L'} ,
    (0,0,1,1,0,0,1):{"digit": 1, "enc": 'L'} ,
    (0,0,1,0,0,1,1):{"digit": 2, "enc": 'L'} ,
    (0,1,1,1,1,0,1):{"digit": 3, "enc": 'L'} ,
    (0,1,0,0,0,1,1):{"digit": 4, "enc": 'L'} ,
    (0,1,1,0,0,0,1):{"digit": 5, "enc": 'L'} ,
    (0,1,0,1,1,1,1):{"digit": 6, "enc": 'L'} ,
    (0,1,1,1,0,1,1):{"digit": 7, "enc": 'L'} ,
    (0,1,1,0,1,1,1):{"digit": 8, "enc": 'L'} ,
    (0,0,0,1,0,1,1):{"digit": 9, "enc": 'L'} ,
    (0,1,0,0,1,1,1):{"digit": 0, "enc": 'G'} ,
    (0,1,1,0,0,1,1):{"digit": 1, "enc": 'G'} ,
    (0,0,1,1,0,1,1):{"digit": 2, "enc": 'G'} ,
    (0,1,0,0,0,0,1):{"digit": 3, "enc": 'G'} ,
    (0,0,1,1,1,0,1):{"digit": 4, "enc": 'G'} ,
    (0,1,1,1,0,0,1):{"digit": 5, "enc": 'G'} ,
    (0,0,0,0,1,0,1):{"digit": 6, "enc": 'G'} ,
    (0,0,1,0,0,0,1):{"digit": 7, "enc": 'G'} ,
    (0,0,0,1,0,0,1):{"digit": 8, "enc": 'G'} ,
    (0,0,1,0,1,1,1):{"digit": 9, "enc": 'G'} ,
    (1,1,1,0,0,1,0):{"digit": 0, "enc": 'R'} ,
    (1,1,0,0,1,1,0):{"digit": 1, "enc": 'R'} ,
    (1,1,0,1,1,0,0):{"digit": 2, "enc": 'R'} ,
    (1,0,0,0,0,1,0):{"digit": 3, "enc": 'R'} ,
    (1,0,1,1,1,0,0):{"digit": 4, "enc": 'R'} ,
    (1,0,0,1,1,1,0):{"digit": 5, "enc": 'R'} ,
    (1,0,1,0,0,0,0):{"digit": 6, "enc": 'R'} ,
    (1,0,0,0,1,0,0):{"digit": 7, "enc": 'R'} ,
    (1,0,0,1,0,0,0):{"digit": 8, "enc": 'R'} ,
    (1,1,1,0,1,0,0):{"digit": 9, "enc": 'R'} 
}

ean13_first_digit_map = {
    ('L','L','L','L','L','L'): 0,
    ('L','L','G','L','G','G'): 1,
    ('L','L','G','G','L','G'): 2,
    ('L','L','G','G','G','L'): 3,
    ('L','G','L','L','G','G'): 4,
    ('L','G','G','L','L','G'): 5,
    ('L','G','G','G','L','L'): 6,
    ('L','G','L','G','L','G'): 7,
    ('L','G','L','G','G','L'): 8,
    ('L','G','G','L','G','L'): 9
}
def decode_ean13( seq: list[int] ) -> tuple[tuple,str]:
    if len(seq) != 97:
        return None
    l_marker = tuple(seq[1:4])
    r_marker = tuple(seq[len(seq)-4:][:3])
    m_marker = tuple(seq[4+7*6:][:5])
    if not (l_marker == r_marker and m_marker == (0,1,0,1,0)):
        return None

    l_part = np.array(seq[4:][:42])
    r_part = np.array(seq[9+42:][:42])

    # il primo gruppo di moduli a sx è sempre di tipo L, popcount dispari
    # in caso contrario, flippa
    if not np.sum(np.resize(l_part,(7))) & 1:
        l_part, r_part = np.flip(r_part), np.flip(l_part)
    
    l_modules = map( lambda el: tuple(el), np.split( l_part, 6 ) ) 
    r_modules = map( lambda el: tuple(el), np.split( r_part, 6 ) )

    try:
        l_decoded = list(map( lambda el: ean13_decmap[el], l_modules ))
        r_decoded = list(map( lambda el: ean13_decmap[el], r_modules ))
    except Exception as e:
        return None
    first_digit_tuple = tuple(map( lambda el: el['enc'], l_decoded ))

    try:
        first_digit = ean13_first_digit_map[first_digit_tuple]
    except Exception as e:
        return None
    l_digits = list(map(lambda el: el['digit'], l_decoded))
    r_digits = list(map(lambda el: el['digit'], r_decoded))
    
    check_digit_parital_sum = reduce(lambda x,y: x+y, [ e * ( 1 if i & 1 else 3 ) for i, e in enumerate(l_digits)])
    check_digit = (10-(check_digit_parital_sum%10))%10
    
    if r_digits[-1] != check_digit:
        return None 

    result_tuple = tuple([first_digit, *l_digits, *r_digits])
    result_str = ''.join(map(lambda x: str(x), result_tuple))
    return result_str, result_tuple

if __name__ == '__main__':
    test = [
       0,1,0,1,0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0
,0,1,0,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,1,0,1,0
,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,1,0
    ]
    print(decode_ean13(test))