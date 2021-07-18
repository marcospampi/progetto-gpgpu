from sys import stderr
import json

def __init__():
    global pattern_id_table
    global code_table
    code_table = []
    table = json.loads(open(__file__.replace('py','json')).read())
    
    pattern_id_table = dict()
    for i,e in enumerate(table):
        e['pattern'] = pattern = tuple(map(lambda x: int(x),tuple(e['pattern'])))
        #e['id'] = id = int(e['id'])
        pattern_id_table[pattern] = i
    for i,e in enumerate(table):
        entry = dict()
        if 'fn' in e:
            entry['isFn'] = True
            entry['fn'] = e['fn']
        else:
            entry['isFn'] = False
            entry['A'] = e['A']
            entry['B'] = e['B']
            entry['C'] = e['C']
        code_table.append(entry)
__init__()

def calculate_checksum( startcode: int, seq: list[int])->bool:
    products = 1

    for i,e in enumerate(seq[:-1]):
        products = products + ( e * ( i + 1 ) )
    checksum = startcode + products
    try:
        checksum = (checksum % 103) ==  seq[-1] + 1
        return True
    except:
        return False

def decode_code128(seq: tuple):
    seq = seq[1:-1]

    start = seq[:11]
    end = seq[-13:]

    if start in pattern_id_table and end in pattern_id_table:
        seq = seq[11:-13]
    else:
        #print( start, end)
        return None
    
    if len(seq) % 11:
        return None
    
    codes = []
    for i in range(int(len(seq)/11)):
        codes.append( pattern_id_table[seq[i*11:i*11+11]] )
    
    if calculate_checksum( pattern_id_table[start], codes) == False:
        return None
    codes = list(map(lambda x: code_table[x], codes))
    mode = code_table[pattern_id_table[start]]['fn'][-1]

    output = []
    for code in codes[:-1]:
        if mode in code:
            value = code[mode]
            if value == "Code A":
                mode = "A"
            elif value == "Code B":
                mode = "B"
            elif value == "Code C":
                mode = "C"
            else:
                output.append(value) 
    
    result_tuple = tuple(output)
    return ''.join(map(lambda x: str(x), result_tuple)), result_tuple 

if __name__ == '__main__':
    sample = (0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0)

    result = decode_code128(sample)
    print(result)
