from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os


NUM_ITER = 5

def get_data(infile):
    count = 0
    data = []

    if not os.path.isfile(infile):
        raise Exception('input file ' + infile + ' does not exist')

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
    nointrin_speedup = []
    intrin_speedup = []
    for iter in range(NUM_ITER):
        nointrin_speedup.append([]);
        intrin_speedup.append([]);
        data = get_data('data/iterations/graph_data_state_vec_size_iter_'+str(iter)+'.txt')
        qs = []

        for d in data:
            qs.append(d['num_qubits'])
            nointrin_speedup[-1].append(d['nointrin']/d['gpu']) #speedup is inverse of time
            intrin_speedup[-1].append(d['intrin']/d['gpu']) #speedup is inverse of time

    #final_nointrin_speedup = (*map(mean,zip(*nointrin_speedup)))
    #final_intrin_speedup = (*map(mean,zip(*intrin_speedup)))
    #final_nointrin_speedup = np.mean(nointrin_speedup, axis=0)
    #final_intrin_speedup = np.mean(intrin_speedup, axis=0)
    final_nointrin_speedup = stats.hmean(nointrin_speedup, axis=0)
    final_intrin_speedup = stats.hmean(intrin_speedup, axis=0)

    #Create figure
    fig, ax1 = plt.subplots()

    #Add titles and labels
    plt.title('gpu speedup vs vector size')
    plt.xlabel('vector size (2^n)', fontweight='bold')
    plt.ylabel('speedup', fontweight='bold')

    #Plot
    ax2 = ax1.twinx()
    ax1.plot(qs, final_intrin_speedup, 'b', label='relative to intrin')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.plot(qs, final_nointrin_speedup, 'r', label='relative to nointrin')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    #Apply linear regression and plot if r2 is sufficiently good
    m,b,r_val,p_val,std_err = stats.linregress(np.array(qs, dtype=float), np.array(final_intrin_speedup, dtype=float))
    if(r_val > 0.98):
        plt.plot(qs, m*np.array(qs, dtype=float)+b, '--k', label='linear interpolation')
        print('gpu speedup relative to intrin vs vec size of 2^n: ', m, b)

    #save plot
    save_filename = 'plots/gpu_speedup_vs_state_vector_size.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
    plt.show()

def plot_time_vs_vec_size_bar():

    intrin_time = []
    nointrin_time = []
    for iter in range(NUM_ITER):
        nointrin_time.append([]);
        intrin_time.append([]);
        data = get_data('data/iterations/graph_data_state_vec_size_iter_'+str(iter)+'.txt')
        qs = []
        final_gpu_time = []
        for d in data:
            qs.append(d['num_qubits'])
            final_gpu_time.append(d['gpu'])
            intrin_time[-1].append(d['intrin'])
            nointrin_time[-1].append(d['nointrin'])
    
    #final_nointrin_time = np.mean(nointrin_time, axis=0)
    #final_intrin_time = np.mean(intrin_time, axis=0)
    final_nointrin_time = np.mean(nointrin_time, axis=0)
    final_intrin_time = np.mean(intrin_time, axis=0)

    final_gpu_time = np.log(final_gpu_time)
    final_nointrin_time = np.log(final_nointrin_time)
    final_intrin_time = np.log(final_intrin_time)

    # set width of bar
    barWidth = 0.25
    
    # set height of bar
    bars1 = final_gpu_time
    bars2 = final_intrin_time
    bars3 = final_nointrin_time
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Make the plot
    fig = plt.figure()
    plt.bar(r1, [x for x in bars1], color='#000000', width=barWidth, edgecolor='white', label='gpu')
    plt.bar(r2, [x for x in bars2], color='#FF6666', width=barWidth, edgecolor='white', label='intrinsics')
    plt.bar(r3, [x for x in bars3], color='#6666FF', width=barWidth, edgecolor='white', label='nonintrinsics')
    
    # Add xticks on the middle of the group bars and show legend
    plt.title('Log scale of time vs size of state vector')
    plt.xlabel('number of qubits', fontweight='bold')
    plt.ylabel('log time (ns)', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], qs)
    plt.legend()
    
    # Save
    save_filename = 'plots/time_vs_state_vector_size.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
    plt.show()

def plot_time_vs_operator_size_bar():

    intrin_time = []
    nointrin_time = []
    for iter in range(NUM_ITER):
        nointrin_time.append([]);
        intrin_time.append([]);
        data = get_data('data/iterations/graph_data_operator_size_iter_'+str(iter)+'.txt')
        qs = []
        final_gpu_time = []
        for d in data:
            qs.append(len(d['qids_list']))
            final_gpu_time.append(d['gpu'])
            intrin_time[-1].append(d['intrin'])
            nointrin_time[-1].append(d['nointrin'])
    
    final_nointrin_time = np.mean(nointrin_time, axis=0)
    final_intrin_time = np.mean(intrin_time, axis=0)
    #final_gpu_time = np.log(final_gpu_time)
    #final_nointrin_time = np.log(final_nointrin_time)
    #final_intrin_time = np.log(final_intrin_time)

    # set width of bar
    barWidth = 0.25
    
    # set height of bar
    bars1 = final_gpu_time
    bars2 = final_intrin_time
    bars3 = final_nointrin_time
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Make the plot
    fig = plt.figure()
    plt.bar(r1, [x / 1e6 for x in bars1], color='#000000', width=barWidth, edgecolor='white', label='gpu')
    #plt.bar(r2, [x / 1e6 for x in bars2], color='#FF6666', width=barWidth, edgecolor='white', label='intrinsics')
    #plt.bar(r3, [x / 1e6 for x in bars3], color='#6666FF', width=barWidth, edgecolor='white', label='nonintrinsics')
    
    # Add xticks on the middle of the group bars and show legend
    plt.title('Time vs operator size')
    plt.xlabel('Number of qubits operated on', fontweight='bold')
    plt.ylabel('time (ms)', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], qs)
    #plt.legend()
    
    # Save
    save_filename = 'plots/time_vs_operator_size.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
    plt.show()

def plot_time_vs_qubit_magnitude_bar():

    intrin_time = []
    nointrin_time = []
    for iter in range(NUM_ITER):
        nointrin_time.append([]);
        intrin_time.append([]);
        data = get_data('data/iterations/graph_data_qubitid_magnitude_iter_'+str(iter)+'.txt')
        qs = []
        final_gpu_time = []
        for d in data:
            qs.append((d['qids_list'][0]))
            final_gpu_time.append(d['gpu'])
            intrin_time[-1].append(d['intrin'])
            nointrin_time[-1].append(d['nointrin'])
    
    final_nointrin_time = np.mean(nointrin_time, axis=0)
    final_intrin_time = np.mean(intrin_time, axis=0)
    #final_gpu_time = np.log(final_gpu_time)
    #final_nointrin_time = np.log(final_nointrin_time)
    #final_intrin_time = np.log(final_intrin_time)

    # set width of bar
    barWidth = 0.25
    
    # set height of bar
    bars1 = final_gpu_time
    bars2 = final_intrin_time
    bars3 = final_nointrin_time
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Make the plot
    fig = plt.figure()
    plt.bar(r1, [x / 1e6 for x in bars1], color='#000000', width=barWidth, edgecolor='white', label='gpu')
    #plt.bar(r2, [x / 1e6 for x in bars2], color='#FF6666', width=barWidth, edgecolor='white', label='intrinsics')
    #plt.bar(r3, [x / 1e6 for x in bars3], color='#6666FF', width=barWidth, edgecolor='white', label='nonintrinsics')
    
    # Add xticks on the middle of the group bars and show legend
    plt.title('Time vs qubit magnitude')
    plt.xlabel('Magnitude of lowest qubit id', fontweight='bold')
    plt.ylabel('time (ms)', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], qs)
    plt.legend()
    
    # Save
    save_filename = 'plots/time_vs_qubit_magnitude.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
    plt.show()

def plot_time_vs_qubit_spacing_bar():

    intrin_time = []
    nointrin_time = []
    for iter in range(NUM_ITER):
        nointrin_time.append([]);
        intrin_time.append([]);
        data = get_data('data/iterations/graph_data_qubitid_spacing_iter_'+str(iter)+'.txt')
        qs = []
        final_gpu_time = []
        for d in data:
            qs.append((d['qids_list'][1]) - d['qids_list'][0])
            final_gpu_time.append(d['gpu'])
            intrin_time[-1].append(d['intrin'])
            nointrin_time[-1].append(d['nointrin'])
    
    final_nointrin_time = np.mean(nointrin_time, axis=0)
    final_intrin_time = np.mean(intrin_time, axis=0)
    #final_gpu_time = np.log(final_gpu_time)
    #final_nointrin_time = np.log(final_nointrin_time)
    #final_intrin_time = np.log(final_intrin_time)

    # set width of bar
    barWidth = 0.25
    
    # set height of bar
    bars1 = final_gpu_time
    bars2 = final_intrin_time
    bars3 = final_nointrin_time
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Make the plot
    fig = plt.figure()
    plt.bar(r1, [x / 1e6 for x in bars1], color='#000000', width=barWidth, edgecolor='white', label='gpu')
    #plt.bar(r2, [x / 1e6 for x in bars2], color='#FF6666', width=barWidth, edgecolor='white', label='intrinsics')
    #plt.bar(r3, [x / 1e6 for x in bars3], color='#6666FF', width=barWidth, edgecolor='white', label='nonintrinsics')
    
    # Add xticks on the middle of the group bars and show legend
    plt.title('Time vs qubit spacing')
    plt.xlabel('Spacing between two qubits with highest at qubit 19', fontweight='bold')
    plt.ylabel('time (ms)', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], qs)
    plt.legend()
    
    # Save
    save_filename = 'plots/time_vs_qubit_spacing.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
    plt.show()

def plot_gpu_speedup_vs_operator_size_line():


    data = []
    nointrin_speedup = []
    intrin_speedup = []

    for iter in range(NUM_ITER):
        nointrin_speedup.append([]);
        intrin_speedup.append([]);
        operator_size = []

        #Split data into lists to plot
        data = get_data('data/iterations/graph_data_operator_size_iter_'+str(iter)+'.txt')
        for d in data:
            operator_size.append(len(d['qids_list']))
            nointrin_speedup[-1].append(d['nointrin']/d['gpu']) #speedup is inverse of time
            intrin_speedup[-1].append(d['intrin']/d['gpu']) #speedup is inverse of time

    #final_nointrin_speedup = np.mean(nointrin_speedup, axis=0)
    #final_intrin_speedup = np.mean(intrin_speedup, axis=0)
    final_nointrin_speedup = stats.hmean(nointrin_speedup, axis=0)
    final_intrin_speedup = stats.hmean(intrin_speedup, axis=0)

    #Create figure
    fig, ax1 = plt.subplots()

    #Add titles and labels
    plt.title('gpu speedup vs number of qubits operated on')
    plt.xticks(operator_size)
    plt.xlabel('qubits operated on', fontweight='bold')
    plt.ylabel('speedup', fontweight='bold')

    #Plot
    ax2 = ax1.twinx()
    ax1.plot(operator_size, final_intrin_speedup, 'b', label='relative to intrin')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.plot(operator_size, final_nointrin_speedup, 'r', label='relative to nointrin')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    #Apply linear regression and plot if r2 is sufficiently good
    m,b,r_val,p_val,std_err = stats.linregress(np.array(operator_size, dtype=float), np.array(final_intrin_speedup, dtype=float))
    if(r_val > 0.98):
        plt.plot(operator_size, m*np.array(operator_size, dtype=float)+b, '--k', label='linear interpolation')
        print('gpu speedup relative to intrin vs vec size of 2^n: ', m, b)

    #save plot
    save_filename = 'plots/gpu_speedup_vs_operator_size.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
    plt.show()

def plot_gpu_speedup_vs_qubit_magnitude_line():


    data = []
    nointrin_speedup = []
    intrin_speedup = []

    for iter in range(NUM_ITER):
        nointrin_speedup.append([]);
        intrin_speedup.append([]);
        lowest_qubit = []

        #Split data into lists to plot
        data = get_data('data/iterations/graph_data_qubitid_magnitude_iter_'+str(iter)+'.txt')
        for d in data:
            lowest_qubit.append((d['qids_list'][0]))
            nointrin_speedup[-1].append(d['nointrin']/d['gpu']) #speedup is inverse of time
            intrin_speedup[-1].append(d['intrin']/d['gpu']) #speedup is inverse of time

    #final_nointrin_speedup = np.mean(nointrin_speedup, axis=0)
    #final_intrin_speedup = np.mean(intrin_speedup, axis=0)
    final_nointrin_speedup = stats.hmean(nointrin_speedup, axis=0)
    final_intrin_speedup = stats.hmean(intrin_speedup, axis=0)

    #Create figure
    fig, ax1 = plt.subplots()

    #Add titles and labels
    plt.title('gpu speedup vs magnitude of 5 consecutive qubits')
    plt.xticks(lowest_qubit)
    plt.xlabel('magnitude of lowest qubit id', fontweight='bold')
    plt.ylabel('speedup', fontweight='bold')
    
    #Plot
    ax2 = ax1.twinx()
    ax1.plot(lowest_qubit, final_intrin_speedup, 'b', label='relative to intrin')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.plot(lowest_qubit, final_nointrin_speedup, 'r', label='relative to nointrin')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    #Apply linear regression and plot if r2 is sufficiently good
    m,b,r_val,p_val,std_err = stats.linregress(np.array(lowest_qubit, dtype=float), np.array(final_intrin_speedup, dtype=float))
    if(r_val > 0.98):
        plt.plot(lowest_qubit, m*np.array(lowest_qubit, dtype=float)+b, '--k', label='linear interpolation')
        print('gpu speedup relative to intrin vs vec size of 2^n: ', m, b)


    #save plot
    save_filename = 'plots/gpu_speedup_vs_qubit_magnitude.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
    plt.show()

def plot_gpu_speedup_vs_qubit_spacing():


    data = []
    nointrin_speedup = []
    intrin_speedup = []

    for iter in range(NUM_ITER):
        nointrin_speedup.append([]);
        intrin_speedup.append([]);
        spacing = []

        #Split data into lists to plot
        data = get_data('data/iterations/graph_data_qubitid_spacing_iter_'+str(iter)+'.txt')
        for d in data:
            spacing.append((d['qids_list'][1]) - d['qids_list'][0])
            nointrin_speedup[-1].append(d['nointrin']/d['gpu']) #speedup is inverse of time
            intrin_speedup[-1].append(d['intrin']/d['gpu']) #speedup is inverse of time

    #final_nointrin_speedup = np.mean(nointrin_speedup, axis=0)
    #final_intrin_speedup = np.mean(intrin_speedup, axis=0)
    final_nointrin_speedup = stats.hmean(nointrin_speedup, axis=0)
    final_intrin_speedup = stats.hmean(intrin_speedup, axis=0)

    #Create figure
    fig, ax1 = plt.subplots()

    #Add titles and labels
    plt.title('gpu speedup vs spacing between 2 qubits for 20-qubit state vector')
    plt.xticks(spacing)
    plt.xlabel('spacing between two qubits with highest at qubit 19', fontweight='bold')
    plt.ylabel('speedup', fontweight='bold')

    #Plot
    ax2 = ax1.twinx()
    ax1.plot(spacing, final_intrin_speedup, 'b', label='relative to intrin')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.plot(spacing, final_nointrin_speedup, 'r', label='relative to nointrin')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    #Apply linear regression and plot if r2 is sufficiently good
    m,b,r_val,p_val,std_err = stats.linregress(np.array(spacing, dtype=float), np.array(final_intrin_speedup, dtype=float))
    if(r_val > 0.98):
        plt.plot(spacing, m*np.array(spacing, dtype=float)+b, '--k', label='linear interpolation')
        print('gpu speedup relative to intrin vs vec size of 2^n: ', m, b)


    #save plot
    save_filename = 'plots/gpu_speedup_vs_qubit_spacing.png'
    plt.savefig(save_filename)
    plt.close(fig)
    print("Generated " + save_filename)
#OLD CODE SNIPPETS
#------------------------------------------------
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
#------------------------------------------------
#END OLD CODE SNIPPETS

def main():
    #Will be writing to plots directory. Make sure it exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    #Will be reading from data directory. Make sure it exists.
    if not os.path.isdir('data'):
        raise Exception('Please create data folder and fill it with the data you want parsed.')

    #Analysis of the effect of vector size
    plot_gpu_speedup_vs_vec_size_line()
    plot_time_vs_vec_size_bar()

    #Analysis of the effect of the operator matrix size (aka number of quibits operated on)
    plot_gpu_speedup_vs_operator_size_line()
    plot_time_vs_operator_size_bar()

    #Analysis of the effect of the magnitude of the qubit ids
    plot_gpu_speedup_vs_qubit_magnitude_line()
    plot_time_vs_qubit_magnitude_bar()

    plot_gpu_speedup_vs_qubit_spacing()
    plot_time_vs_qubit_spacing_bar()

if __name__ == "__main__":
    main()
