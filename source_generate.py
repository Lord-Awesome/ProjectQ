import sys
import random

n = int(sys.argv[1])
with open(f'source_matrix.txt', 'w') as f:
	for i in range(2**n):
		for j in range(2**n):
			#f.write(f'({str(random.random())}, {str(random.random())})')
			if(i == j):
				f.write('(1.0,0.0)\n')
			else:
				f.write('(0.0,1.0)\n')
			#f.write("(0,0)")
