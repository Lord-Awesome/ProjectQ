import sys
import random

n = int(sys.argv[1])
with open(f'matrix{n}.txt', 'w') as f:
    for _ in range(2**n):
        for _ in range(2**n):
            f.write(f'({str(random.random())}, {str(random.random())})')