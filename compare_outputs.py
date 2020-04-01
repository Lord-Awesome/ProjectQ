import math
rms = 0
i = 0
with open('output.txt', 'r') as f1:
	with open('output_truth.txt', 'r') as f2:
		for line1, line2 in zip(f1, f2):
			i += 1
			if line1 != line2:
				tup1 = line1.replace('(','').replace(')','').replace('\n','').split(',')
				tup1[0] = float(tup1[0])
				tup1[1] = float(tup1[1])
				tup2 = line2.replace('(','').replace(')','').replace('\n','').split(',')
				tup2[0] = float(tup2[0])
				tup2[1] = float(tup2[1])
				rms += (tup1[0] - tup2[0]) ** 2 + (tup1[0] - tup2[0]) ** 2
				if not math.isclose(tup1[0], tup2[0], rel_tol=1E-4, abs_tol=1E-4) or not math.isclose(tup1[1], tup2[1], rel_tol=1E-4, abs_tol=1E-4):
					print(i, tup1, tup2)
					exit(1)

print(math.sqrt(rms / i))
