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
    df2 = pd.read_sql('results_dmota', db)
    df3 = pd.read_sql('experiments', db)
    return df2, df3

def grab_run(db, index):
    runs = pd.read_sql("results_dmota", con=db)
    runs = runs.sort_values(by='index')
    #print(runs)
    run= runs.iloc[[index]]
    print(f"fetched run {index}")
    return run


def get_critical_node_indexes(nDims = 9):
    indexes=[]
    for i in range(nDims):
        indexes.append(int(nDims/2)+nDims*i)
    return indexes

def save_to_db(db, lifetime, latency, precentage_missed, nmissed, algorithm, nnodes, ntasks, predictor, static, index, settings ):
    import numpy as np
    
    results = {'index' : index,
                'lifetime' : lifetime,
                'latency' : latency,
                'percentage_missed' : percentage_missed,
                'nMissed' : nMissed,
               'algorithm' : algorithm,
               'settings' : json.dumps(settings),
               'nnodes' : nnodes,
               'ntasks' : ntasks,
               'static' : static,
               'predictor' : predictor
               }
    #np.random.seed(4356)
    df = pd.DataFrame(results, index=[index])
    df.set_index('index', inplace=True)
    df.to_sql('results_final_dmota', db, if_exists='append')

    #print(lifetime)
    #print(latency)
    #print(nMissed)
    #print(missed_sequence)



def sim_run(bests, archives, elitesets, **kwargs):
    import network
    import ns.core
    time = 0
    latency = 0
    nNodes = kwargs['nNodes']
    alg = kwargs['algorithm']
    network_creator = topologies.network_topologies[kwargs['network_creator']]
    nTasks = kwargs['nTasks']
    task_creator = topologies.task_topologies[kwargs['task_creator']]
    energy_list = kwargs['energy_list_sim']
    networkGraph = network_creator(**kwargs)
    taskGraph = task_creator(networkGraph, **kwargs)   
    posL = [[float(node.posx), float(node.posy)] for node in networkGraph.nodes()]
    kwargs.update({'posList' : posL})
    try:
        net = network.Network(networkGraph, **kwargs)
    except Exception as e:
        print(f"Error during network creation: {e}")
        raise e
    network.createTasksFromGraph(net, taskGraph, bests[0], **kwargs)
    latency_list = []
    received_list = []
    actrcvd = []
    sendsent = []
    sent_list = []
    send_list = []
    energy_list = []
    node_status = []
    node_pos = []
    node_broadcast = []
    act_list = []
    seqNumTx=[]
    seqNumRx=[]
    time = []
    def getTime(time = []):
        time.append(ns.core.Simulator.Now().GetSeconds())    
    ns.core.RngSeedManager.SetRun(kwargs['run_number'])
    a_fixed=[]
    dt = 30
    for a in bests[1:]:
        for i, alloc in enumerate(a):
            a_fixed.append((i,[alloc]))
        ns.core.Simulator.Schedule(ns.core.Seconds(dt), net.controlTask.Reallocate, a_fixed)
        dt += 30
    if elitesets[0] == elitesets:
        elitesets = elitesets[1:]
    print(len(archives))
    print(len(bests))
    print(len(elitesets))
    main_archive = []
    for a in archives:
        new_archive = []
        for ai in a:
            a_fixed = [] 
            for i, alloc in enumerate(ai):
                a_fixed.append((i,[alloc]))
            new_archive.append(a_fixed)
        main_archive.append(new_archive)
    
    main_eliteset = []
    for e in elitesets:
        new_eliteset = []
        for ei in e:
            e_fixed = [] 
            for i, alloc in enumerate(ei):
                e_fixed.append((i,[alloc]))
            new_eliteset.append(e_fixed)
        main_eliteset.append(new_eliteset)
    
    dt = 0
    if len(main_eliteset) > 0:
        for a,e in zip (main_archive, main_eliteset):
            ea = e+a
            ns.core.Simulator.Schedule(ns.core.Seconds(dt), net.controlTask.SetArchive, ea)
            dt += 30
    else:
        for a in main_archive:
            ns.core.Simulator.Schedule(ns.core.Seconds(dt), net.controlTask.SetArchive, a)

    ns.core.Simulator.ScheduleDestroy(net.getPackagesSent, sent_list, send_list, seqNumTx)
    ns.core.Simulator.ScheduleDestroy(net.getPackagesReceived, received_list, act_list, seqNumRx)
    ns.core.Simulator.ScheduleDestroy(net.getEnergy, energy_list)
    ns.core.Simulator.ScheduleDestroy(getTime, time)
    ns.core.Simulator.ScheduleDestroy(net.getNodeStatus, node_status)
    ns.core.Simulator.ScheduleDestroy(net.getLatency, latency_list)
    ns.core.Simulator.Stop(ns.core.Time(ns.core.Seconds(1200)))
    print("running sim")
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()
    print("sim finished")
    
    lifetime = time[0]
    latency = max(latency_list) if len(latency_list) > 0 else 98999

    missed_packages = sent_list[0] - received_list[0]
    percentage_missed = missed_packages/sent_list[0] if sent_list[0] > 0 else 1


    return -lifetime, latency, percentage_missed, missed_packages
    
    














    net.controlTask = 0
    net = 0
    nodetasks = 0
    





if __name__ == "__main__":
    import pickle as pck
    import argparse
    import sqlalchemy as sql
    db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
    import json
    import ast
    import os
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', action='store', default=-1)
    args = parser.parse_args()
    if int(args.index) > 0:
        jobid = int(args.index)
    #data, data_exp = grab_data(db)
    row = grab_run(db, jobid)
    settings = json.loads(row.iloc[0]['settings'])
    bests = pck.loads(row.iloc[0]['bests'])
    fronts = pck.loads(row.iloc[0]['fronts'])
    archives = pck.loads(row.iloc[0]['archives'])
    elitesets = pck.loads(row.iloc[0]['elitesets'])
    algorithm = settings['algorithm']
    nNodes = settings['nNodes']
    nTasks = settings['nTasks']
    static = settings['static']
    predictor = settings['predictor']
    print(algorithm)
    lifetime, latency, percentage_missed, nMissed = sim_run(bests, archives, elitesets, **settings)
    print(lifetime)    
    print(latency)    
    print(percentage_missed)    
    print(nMissed)    
        

    save_to_db(db, lifetime, latency, percentage_missed, nMissed, algorithm, nNodes, nTasks, predictor, static, jobid, settings)







