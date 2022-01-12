import json
import pickle
from deap import creator, base
from individual import ListWithAttributes
import math
from collections import defaultdict
import pandas as pd
from individual import ListWithAttributes
from deap import tools
import topologies
import network
import itertools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0,))
creator.create("Individual", ListWithAttributes, fitness=creator.FitnessMin)

def sortEpsilonNondominated(individuals, k, first_front_only=False):
    if k == 0:
        return []
    angle = 30
    a = math.tan(math.radians(angle*2))/2
    map_fit_ind = defaultdict(list)
    max_fit = [0,0,0]
    for ind in individuals:
        for i,fit in enumerate(ind.fitness.values):
          if abs(fit) > max_fit[i]:
            max_fit[i] = abs(fit)
    for ind in individuals:
        new_fit = creator.FitnessMin((ind.fitness.values[0]/max_fit[0]*1+ind.fitness.values[1]/max_fit[1]*a + ind.fitness.values[2]/max_fit[2]*a, ind.fitness.values[1]/max_fit[1]*1+ind.fitness.values[0]/max_fit[0]*a+ ind.fitness.values[2]/max_fit[2]*a, ind.fitness.values[2]/max_fit[2]*1 + ind.fitness.values[0]/max_fit[0]*a + ind.fitness.values[1]/max_fit[1]*a,))
        map_fit_ind[new_fit].append(ind)
    fits = map_fit_ind.keys()

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in list(fits)[i+1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

def grab_data(db):
    df2 = pd.read_sql('results_mobility', db)
    df3 = pd.read_sql('experiments', db)
    return df2, df3


def get_critical_node_indexes(nDims = 9):
    indexes=[]
    for i in range(nDims):
        indexes.append(int(nDims/2)+nDims*i)
    return indexes

def save_to_db(db, lifetime, latency, missed_sequence, algorithm, nnodes, ntasks, predictor, static, index ):
    import numpy as np
    
    results = {'index' : index,
                'lifetime' : lifetime,
                'latency' : latency,
                'nMissed' : missed_sequence,
               'algorithm' : algorithm,
               'nnodes' : nnodes,
               'ntasks' : ntasks,
               'static' : static,
               'predictor' : predictor
               }
    #np.random.seed(4356)
    df = pd.DataFrame(results, index=[index])
    df.set_index('index', inplace=True)
    df.to_sql('results_final_mobility', db, if_exists='append')

    #print(lifetime)
    #print(latency)
    #print(nMissed)
    #print(missed_sequence)



def sim_run(allocation_series = [], stopTime = 600, **kwargs):
    import ns.core
    import ns.network
    import ns.point_to_point
    import ns.applications
    import ns.wifi
    import ns.lr_wpan
    import ns.mobility
    import ns.csma
    import ns.internet 
    import ns.sixlowpan
    import ns.internet_apps
    import ns.energy
    time = 0
    latency = 99999
    nNodes = kwargs['nNodes']
    nTasks = kwargs['nTasks']
    network_creator = topologies.network_topologies[kwargs['network_creator']]
    task_creator = topologies.task_topologies[kwargs['task_creator']]
    energy_list = [100]*nNodes
    networkGraph = network_creator(**kwargs)
    posL = [[float(node.posx), float(node.posy)] for node in networkGraph.nodes()]
    kwargs['posList'] = posL
    taskGraph = task_creator(networkGraph, **kwargs)   
    try:
        net = network.Network(networkGraph, **kwargs)
    except Exception as e:
        print(f"Error during network creation: {e}")
        raise e
    network.createTasksFromGraph(net, taskGraph, allocation_series[0][0], **kwargs)
    node_status = []
    latency_list = []
    received_list = []
    actrcvd = []
    sendsent = []
    sent_list = []
    send_list = []
    energy_list = []
    node_status = []
    act_list = []
    seqNumTx=[]
    seqNumRx=[]
    time = []
    def getTime(time = []):
        time.append(ns.core.Simulator.Now().GetSeconds())    
    ns.core.RngSeedManager.SetRun(kwargs['run_number'])
    for a in allocation_series[1:]:
        a_fixed = [(0,[0])]
        for i, alloc in enumerate(a[0]):
            a_fixed.append((i+1,[alloc]))
        #print(f"scheduling realloc to {a_fixed} at time {a[1]}")
        ns.core.Simulator.Schedule(ns.core.Seconds(a[1]), net.controlTask.Reallocate, a_fixed, net.taskApps)
    ns.core.Simulator.ScheduleDestroy(net.getPackagesSent, sent_list, send_list, seqNumTx)
    ns.core.Simulator.ScheduleDestroy(net.getPackagesReceived, received_list, act_list, seqNumRx)
    ns.core.Simulator.ScheduleDestroy(net.getEnergy, energy_list)
    ns.core.Simulator.ScheduleDestroy(getTime, time)
    ns.core.Simulator.ScheduleDestroy(net.getNodeStatus, node_status)
    ns.core.Simulator.ScheduleDestroy(net.getLatency, latency_list)
    ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(150)))
    print("running sim")
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()
    print("sim finished")
    latency = max(latency_list) if len(latency_list) > 0 else latency
    
    nMissed = 0
    missed_sequence = []
    for tx in itertools.zip_longest(*seqNumTx,fillvalue=-1):
        found = [False for x in tx]
        for i,a in enumerate(tx):
            for r in seqNumRx:
                if a in r or a == -1:
                    found[i] = True
                    continue
        if not all(found):
            nMissed +=1
            missed_sequence.append(1)
        else:
            missed_sequence.append(0)
    
    start_energy = kwargs['energy_list']
    #the energy actually left on the nodes
    energy_sim = energy_list
    delta_energy = []
    lifetimes = []
    

    for old_en, new_en, in zip(start_energy, energy_sim):
        #how much energy was spent?
        deltaE = (old_en-new_en)/time[0]
        delta_energy.append(deltaE)
        if deltaE > 0:
            # how long will the energy last on the actual network?
            lifetimes.append(old_en/deltaE)
    critnodes = get_critical_node_indexes()
    crit_lifetimes = []
    for i in critnodes:
        crit_lifetimes.append(lifetimes[i])
    lifetime = min(crit_lifetimes)
    time = time[0]
    
    return -lifetime, latency, nMissed, missed_sequence, time
    
    














    net.controlTask = 0
    net = 0
    nodetasks = 0
    





if __name__ == "__main__":
    import pickle as pck
    import sqlalchemy as sql
    db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
    import json
    import ast
    data, data_exp = grab_data(db)
    for index, row in data.iterrows():
        settings = json.loads(row['settings'])
        settings.update({'posList' : []})
        settings.update({'network_status' : []})
        settings.update({'energy_list_sim' : []})
        bests = pck.loads(row['bests'])
        fronts = pck.loads(row['fronts'])
        datap = f"{settings['datapath'][:-5]}_fronts.pck"
        algorithm = settings['algorithm']
        nnodes = settings['nNodes']
        ntasks = settings['nTasks']
        static = settings['static']
        predictor= settings['predictor']
        nonConeDomFronts = []
            #print(fronts)
            #print(datap)
        deltaT = 30
        index = 0
        final_front = []
        if len(fronts) < 20:
            stopTime = 30*len(fronts)
        if len(fronts) > 1:
            for x in fronts:
                trimmed_front = [y for y in x if y.fitness.values[1] < 9000]
                if len(trimmed_front) > 0:
                    nonConeDomFront = sortEpsilonNondominated(trimmed_front,len(trimmed_front))[0]
                    #for j in nonConeDomFront[0].fitness.values:
                        #print(j)
                    final_front.append((tools.selBest(nonConeDomFront,1)[0],index*deltaT))
                index += 1
            #lifetime, latency, nMissed, missed_sequence, time = sim_run(final_front, **settings)
            lifetime = 5000
            latency = -1
            nMissedSeq = []
            nMissed = 0
            for x in final_front:
                lifetime = min(lifetime, x[0].fitness.values[0])
                latency = max(latency, x[0].fitness.values[1])
                nMissed +=  x[0].fitness.values[2]/20
            if len(final_front) < 20:
                nMissed = nMissed/len(final_front)*19
            
            save_to_db(db, lifetime, latency, nMissed, algorithm, nnodes, ntasks, predictor, static, index)







