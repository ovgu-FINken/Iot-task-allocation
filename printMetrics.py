import pandas as pd
import seaborn as sns
import pickle as pck
import matplotlib.pyplot as plt
import argparse
import os

def min2digits(a):
    s = str(a)
    if len(s) < 2:
      s = "0" + str(a)
    return s


def get_data(alg = 'nsga2', net = 'Grid', task = 'EncodeDecode', nNodes = 81, nTasks = 19):
    filepath = f"results/{alg}/{net}/{task}/objectives_nodes{nNodes}_tasks{nTasks}"
    data = []
    for i in range(31):
        objectives = []
        for j in range(5):
            try:
                with open(f"{filepath}_{min2digits(i)}_{min2digits(j)}.pck", "rb") as f:
                    objectives = pck.load(f)
            except FileNotFoundError as e:
                break
        NL = 0
        Latency = 0
        if len(objectives) == 0:
            break
        for i in objectives:
            NL += -abs(i[0])
            Latency = max(Latency, i[1]) if i[1] < 99999 else Latency
        if alg == 'nsga2':
            algname = "NSGAII"
        elif alg == "dtas":
            algname = "DTAS"
        df_entry = (algname, net, task, nNodes, nTasks, NL, Latency/1000)
        data.append(df_entry)
    return data

def makeplot(x, y, hue, data, net=None, dodge = False, widths = 0.3):
    kwargs = {'widths' : widths}
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data[data['net'] == net])
    
    filepath = f"results/plots/"
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i,label in enumerate(labels):
        if label == "EncodeDecode":
            labels[i] = "Sink-Source"
        elif label == "TwoTaskWithProcessing":
            labels[i] = "Line"
    ax.set_xticklabels(labels)
    if y == 'Latency':
        ax.set_yscale("log")
    plt.xlabel(f"Task Setup")
    plt.ylabel(f"{y} (s)")
    plt.title(f"{net}_{task}")
    plt.savefig(f"{filepath}NSGAIIvDTAS_{net}_{y}")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default = "Grid")
    parser.add_argument('--task', type=str, default = 'EncodeDecode')
    parser.add_argument('--alg', type=str, default = 'nsga2')
    parser.add_argument('--nodes', type=int, default = 81)
    parser.add_argument('--tasks', type=int, default = 19)
    parser.add_argument('--x', type=str, default ="algorithm")
    parser.add_argument('--y', type=str, default ="NL")

    args = parser.parse_args()
    

    algorithms = ['nsga2', 'dtas']
    networks = ['Grid', 'Line']
    tasks = ['EncodeDecode', 'OneSink', 'TwoTaskWithProcessing']

    if args.net == "Grid":
        args.nodes = 81 if args.nodes > 20 else 16
    if args.task == "EncodeDecode":
        args.tasks = 19

    data = get_data()
    data2 = get_data(alg='dtas')
    df1 = pd.DataFrame(data, columns=['algorithm', 'net', 'task', 'nNodes', 'nTasks', 'NL', 'Latency'])
    df2 = pd.DataFrame(data2, columns=['algorithm', 'net', 'task', 'nNodes', 'nTasks', 'NL', 'Latency'])
    
    dataframes = []
    for net in networks:
        for algo in algorithms:
            for task in tasks:
                if net == "Grid":
                    nodes = 81
                else:
                    nodes = 20
                if task == "EncodeDecode":
                    nTasks = 19
                elif task == "OneSink":
                    nTasks = 11
                else: 
                    nTasks = 20
                data = get_data(alg=algo, net = net, task = task, nNodes = nodes, nTasks = nTasks)
                df1 = pd.DataFrame(data, columns=['algorithm', 'net', 'task', 'nNodes', 'nTasks', 'NL', 'Latency'])
                dataframes.append(df1)
    
    df = pd.concat(dataframes)
    print(df.to_string())
    for net in networks:
        makeplot('task', 'NL', 'algorithm', df, net= net)
        makeplot('task', 'Latency', 'algorithm', df, net= net)

