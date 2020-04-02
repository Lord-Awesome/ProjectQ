import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def get_data(infile):
    count = 0
    data = []
    with open(infile, 'r') as f:
        for line in f:
            count += 1
    
    if count % 4 != 0 or count == 0:
        raise Exception('malformed input file')

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
            
def plot_gpu_speedup_vs_vec_size_line():
    data = get_data('graph_data_state_vec_size.txt')
    qs = []
    nointrin_speedup = []
    intrin_speedup = []

    for d in data:
        qs.append(d['num_qubits'])
        nointrin_speedup.append(d['nointrin']/d['gpu']) #speedup is inverse of time
        intrin_speedup.append(d['intrin']/d['gpu']) #speedup is inverse of time

    plt.plot(qs, intrin_speedup, 'b', label='relative to intrin')

    m,b,r_val,p_val,std_err = stats.linregress(np.array(qs, dtype=float), np.array(intrin_speedup, dtype=float))
    if(r_val > 0.98):
        plt.plot(qs, m*np.array(qs, dtype=float)+b, '--k', label='linear interpolation')
        print('gpu speedup relative to intrin vs vec size of 2^n: ', m, b)

    plt.title('gpu speedup vs vector size')
    plt.xlabel('vector size (2^n)', fontweight='bold')
    plt.ylabel('speedup', fontweight='bold')

    plt.legend()
    plt.savefig('plots/gpu_speedup_vs_state_vector_size.png')
    plt.show()

def plot_time_vs_vec_size_bar():
    data = get_data('graph_data_state_vec_size.txt')
    qs = []
    for d in data:
        qs.append(d['num_qubits'])
        gpu_time.append(d['gpu'])
        intrin_time.append(d['intrin'])
        nointrin_time.append(d['nointrin'])

    # set width of bar
    barWidth = 0.25
    
    # set height of bar
    bars1 = gpu_time
    bars2 = intrin_time
    bars3 = nointrin_time
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Make the plot
    plt.bar(r1, bars1, color='#FFFFFF', width=barWidth, edgecolor='white', label='gpu')
    plt.bar(r2, bars2, color='#FF6666', width=barWidth, edgecolor='white', label='intrinsics')
    plt.bar(r3, bars3, color='#6666FF', width=barWidth, edgecolor='white', label='nonintrinsics')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('number of qubits', fontweight='bold')
    plt.ylabel('time (ns)', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], qs)
    
    # Create legend & Show graphic
    plt.legend()
    plt.savefig('plots/time_vs_state_vector_size.png')
    plt.show()


#operator_size.append(len(d['qids_list']))

#print('num_qubits: ' + d['num_qubits'] + ' speedup over nointrin: ' + str(d['nointrin']/d['gpu']) + ' speedup over intrin: ' + str(d['intrin']/d['gpu']))


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

# # set width of bar
# barWidth = 0.25
 # 
# # set height of bar
# bars1 = [1] * len(data)
# bars2 = intrin_speedup
# bars3 = nointrin_speedup
 # 
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
 # 
# # Make the plot
# #plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
# plt.bar(r2, bars2, color='#FF6666', width=barWidth, edgecolor='white', label='intrinsics')
# plt.bar(r3, bars3, color='#6666FF', width=barWidth, edgecolor='white', label='nonintrinsics')
 # 
# # Add xticks on the middle of the group bars
# plt.xlabel('number of qubits', fontweight='bold')
# plt.ylabel('speedup over nointrin', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], operator_size)
 # 
# # Create legend & Show graphic
# plt.legend()
# plt.savefig("test2.png")
# plt.show()

def main():
    #Will be writing to plots directory. Make sure it exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    #Will be reading from data directory. Make sure it exists.
    if not os.path.isdir('data')
        raise Exception('Please create data folder and fill it with the data you want parsed.')

    #Analysis of the effect of vector size
    plot_gpu_speedup_vs_vec_size_line()