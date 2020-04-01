
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def get_data(infile):
    count = 0
    data = []
    with open(infile, 'r') as f:
        for line in f:
            count += 1
    
    if count % 4 != 0 or count == 0:
        raise exception('malformed input file')

    with open(infile, 'r') as f: 
        for _ in range(int(count / 4)):
            qids_split = f.readline().split(' ')
            num_qubits = None
            qids_list = []
            for i, qid in enumerate(qids_split):
                if i % 2:
                    if i == 1:
                        num_qubits = qid
                    else:
                        qids_list.append(int(qid.replace('\n','')))
            qids_list.sort()
            
            nointrin = int(f.readline().split(':')[1])
            intrin = int(f.readline().split(':')[1])
            gpu = int(f.readline().split(':')[1])
            data.append({
                'num_qubits' : num_qubits,
                'qids_list' : qids_list,
                'nointrin' : nointrin,
                'intrin' : intrin,
                'gpu' : gpu
            })
    return data
            
data = get_data('time_comparison.txt')


def get_linear

qs = []
nointrin_speedup = []
intrin_speedup = []

operator_size = []

for d in data:
    #print('num_qubits: ' + d['num_qubits'] + ' speedup over nointrin: ' + str(d['nointrin']/d['gpu']) + ' speedup over intrin: ' + str(d['intrin']/d['gpu']))
    qs.append(d['num_qubits'])
    operator_size.append(len(d['qids_list']))
    nointrin_speedup.append(d['nointrin']/d['gpu'])
    intrin_speedup.append(d['intrin']/d['gpu'])

    print('')
    #intrin_over_nointrin.append(d['nointrin']/d['intrin'])


# plt.plot(qs, nointrin_speedup, 'r', label='nointrin')
# plt.plot(qs, intrin_speedup, 'b', label='intrin')
# #plt.plot(qs, intrin_over_nointrin, 'g', label='test')

# # m,b = np.polyfit(np.array(qs), np.array(nointrin_speedup), 1)
# # print('nointrin: ', m, b)
# # m,b = np.polyfit(np.array(qs), np.array(intrin_speedup), 1)
# # print('intrin: ', m, b)

# gradient, intercept, r_value, p_value, std_err = stats.linregress(np.array(qs, dtype=float), np.array(nointrin_speedup,dtype=float))
# print(gradient, intercept, r_value, p_value, std_err)
# gradient, intercept, r_value, p_value, std_err = stats.linregress(np.array(qs, dtype=float), np.array(intrin_speedup,dtype=float))
# print(gradient, intercept, r_value, p_value, std_err)

# plt.xlabel('number of qubits')
# plt.ylabel('speedup')
# plt.title('#qubits vs speedup')
# plt.legend()
# plt.grid(True)
# plt.savefig("test.png")
# plt.show()

# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [1] * len(data)
bars2 = intrin_speedup
bars3 = nointrin_speedup
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
#plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, bars2, color='#FF6666', width=barWidth, edgecolor='white', label='intrinsics')
plt.bar(r3, bars3, color='#6666FF', width=barWidth, edgecolor='white', label='nonintrinsics')
 
# Add xticks on the middle of the group bars
plt.xlabel('number of qubits', fontweight='bold')
plt.ylabel('speedup over nointrin', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], operator_size)
 
# Create legend & Show graphic
plt.legend()
plt.savefig("test2.png")
plt.show()
