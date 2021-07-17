import argparse
from decoders.ean13 import decode_ean13
from decoders.code128 import decode_code128

if __name__ == '__main__':
    arpa = argparse.ArgumentParser()
    arpa.add_argument("source", help="Sequenze da verificare")

    filename = arpa.parse_args().source

    seqs = dict()

    with open(filename) as file:
        for row in file.readlines():
            row = row.split(' ')[:-1]
            row = tuple(map(lambda e: int(e), row))
            count = seqs[row] if row in seqs else 0
            seqs[row] = count + 1
    decoded = [  ]

    for key in seqs:
        tests = [
            decode_code128(key),
            decode_ean13(key)
        ]
        for i in tests:
            if i != None:
                decoded.append(i)

    print("Trovati {0} codici:".format(len(decoded)))
    for e in decoded:
        print(*e)
                

