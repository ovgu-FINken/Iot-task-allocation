import sqlalchemy as sql
import pandas as pd
import json
import argparse
from enum import IntEnum

class JobStatus(IntEnum):
    TODO = 0
    IN_PROGRESS = 1
    DONE = 2
    FAILED = 3

def ceate_experiments_mmota(task_numbers, settings):
    with open(f"datasets/{'static' if static else 'mobile'}/{nNodes}/{predictor if not static else ''}/settings_{i}.json") as f:
        settings = json.load(f)
        crossover = 'nsga2'
        NGEN = 50
        settings.update({'crossover' : crossover})
        settings.update({'NGEN' : NGEN})
        for nTasks in task_numbers:
            settings.update({'nTasks' : nTasks})
            settings.update({'experiment' : 'mmota'})
            settings.update({'task_creator' : task_creator})
            settings.update({'algorithm' : 'mmota'})
            settings.update({'NGEN_realloc' : 20})
            settings.update({'predictor' : predictor})
            old_results = pd.read_sql("experiments", con=db)
            min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
            datapath = f"datasets/{'static' if static else 'mobile'}/{nNodes}/{predictor if not static else ''}/positions_{i}.json"
            if static:
                predpath = f"datasets/static/{nNodes}/positions_{i}.json"
            else:
                predpath = f"datasets/mobile/{nNodes}/{predictor if not static else ''}/predictions_{i}.json"
            settings.update({'datapath' : datapath})
            settings.update({'predpath' : predpath})
            run = {'index' : min_index,
               'experiment' : 'mmota',
               'algorithm' : algorithm,
               'status' : JobStatus.TODO,
               'settings' : json.dumps(settings),
               'datapath' : datapath
                }
    return run, settings

def create_experiments_mota(settings):
    crossover = 'nsga2'
    NGEN = 100
    settings.update({'predictor' : 'perfect'})
    settings.update({'crossover' : crossover})
    settings.update({'NGEN' : NGEN})
    settings.update({'experiment' : 'surrogates'})
    old_results = pd.read_sql("experiments", con=db)
    min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
    datapath = f"datasets/surrogates/{settings['eval_mode']}/{settings['nNodes']}/{settings['nTasks']}/.json"
    settings.update({'datapath' : datapath})
    run = {'index' : min_index,
       'experiment' : 'surrogates',
       'algorithm' : settings['algorithm'],
       'eval_mode': settings['eval_mode'],
       'status' : JobStatus.TODO,
       'settings' : json.dumps(settings),
       'datapath' : datapath
        }
    return run, settings
 
def create_experiments_amota(settings):
    crossover = 'nsga2'
    NGEN = 20
    NGEN_realloc = 10
    settings.update({'predictor' : 'perfect'})
    settings.update({'crossover' : crossover})
    settings.update({'NGEN' : NGEN})
    settings.update({'NGEN_realloc' : NGEN_realloc})
    settings.update({'experiment' : 'error_rate'})
    old_results = pd.read_sql("experiments", con=db)
    datapath = f"datasets/errorates/{settings['eval_mode']}/{settings['error_shape']}/{settings['error_dur_shape']}/.json"
    min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
    run = {'index' : min_index,
       'experiment' : 'errorrate',
       'algorithm' : settings['algorithm'],
       'eval_mode': settings['eval_mode'],
       'status' : JobStatus.TODO,
       'settings' : json.dumps(settings),
       'datapath' : datapath
        }
    return run, settings


def create_experiments_from_files():
    import glob
    nTasks = 25
    algorithm = 'dmota'
    settings_paths = glob.glob("/home/repos/Iot-task-allocation/datasets/**/*settings*", recursive=True)
    for spath in settings_paths:
        with open(spath, 'rb') as sfile:
            settings = json.load(sfile)
            statuspath = spath.replace("settings", "status")
            pospath = spath.replace("settings", "positions")
            predpath = spath.replace("settings", "predictions")
            broadcastpath = spath.replace("settings", "broadcast")
            with open(statuspath) as f:
                status = json.load(f)
                settings.update({'network_status_all' : status})
            with open(pospath) as f:
                status = json.load(f)
                settings.update({'posList_all' : status})
            with open(predpath) as f:
                status = json.load(f)
                settings.update({'prediction_data_all' : status})
            with open(broadcastpath) as f:
                status = json.load(f)
                settings.update({'broadcast_status_all' : status})
            for algorithm in ['dmota', 'amota', 'mmota']:
                settings.update({'algorithm' : algorithm})
                settings.update({'nTasks' : 25})
                settings.update({'experiment' : 'dmota'})
                settings.update({'popSize' : 100})
                old_results = pd.read_sql("experiments", con=db)
                min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
                run = {'index' : min_index,
                    'experiment' : 'dmota',
                    'algorithm' : settings['algorithm'],
                    'eval_mode': settings['eval_mode'],
                    'status' : JobStatus.TODO,
                    'settings' : json.dumps(settings),
                    }
                df = pd.DataFrame(run, index=[0])
                df.set_index('index', inplace=True)
                df.to_sql('experiments', db, if_exists='append')
    


def create_experiments():
    task_numbers = [10]
    diml = [5]
    algorithms = ['amota', 'mmota', 'dmota']
    eval_modes = ['sim']#, 'surrogate']
    network_creator = 'Manhattan'
    #task_creator = 'EncodeDecode'
    task_creator = 'OneSink'
    mobl = [0,20,40]
    errors_enabled = True
    error_settings = [[250,250], [250,60],[1000,250],[1000,60]] if errors_enabled else [[1,1]]
    for err_params in error_settings:
        for static in [True]:
            predl = ['perfect', 'target'] if not static else ['perfect']
            for predictor in predl:
                print(f"creating {predictor} predictor data")
                mobl = [0,30,40]
                mobl = mobl if not static else [0]
                for dims in diml:
                    print(f"{dims} dim data")
                    for nMob in mobl:
                        mobileNodes = nMob
                        nNodes = dims**2 + mobileNodes
                        for algorithm in algorithms:
                            for eval_mode in eval_modes:
                                for nTasks in task_numbers:
                                    surrogates = ['gnn','graph'] if eval_mode =='surrogate' else ['None']
                                    for surrogate in surrogates:
                                        for i in range(11):
                                            run = None
                                            if algorithm == 'mmota':
                                                run, settings = create_experiments_mmota(task_numbers)
                                            else:
                                                settings = {'nNodes' : nNodes,
                                                     'mobileNodeCount' : mobileNodes,
                                                     'network_creator' : network_creator,
                                                     'algorithm' : algorithm,
                                                     'dimx' : dims,
                                                     'dimy' : dims,
                                                     'nTasks' : nTasks,
                                                     'deltax' :100,
                                                     'deltay': 100,
                                                     'surrogate' : surrogate,
                                                     'task_creator' : task_creator,
                                                     'energy_list' : [500]*nNodes ,
                                                     'posList' : [],
                                                     'init_energy' : 500,
                                                     'verbose' : False,
                                                     'capture_packets' : False,
                                                     'pcap_filename' : f"pcap",
                                                     'enable_errors' : errors_enabled,
                                                     'seed' : 3141 + i*21,
                                                     'error_shape' : 1.0,
                                                     'error_dur_shape' : 1.0,
                                                     'error_scale' : err_params[0],
                                                     'error_dur_scale' : err_params[1],
                                                     'broadcast_status': [0]*nNodes,
                                                     'network_status' : [[1,0]]*nNodes,
                                                     'routing' : True,
                                                     'static' : static,
                                                     'run_number' : i,
                                                     'eval_mode' : eval_mode
                                                     }
                                                if algorithm =='nsga2':
                                                    run, settings = create_experiments_mota(settings)
                                                elif algorithm == 'amota':
                                                    run, settings = create_experiments_amota(settings)
                                            df = pd.DataFrame(run, index=[0])
                                            df.set_index('index', inplace=True)
                                            df.to_sql('experiments', db, if_exists='append')

                

def fetch_run(index, db):
    runs = pd.read_sql("experiments", con=db)
    runs = runs.sort_values(by='index')
    #print(runs)
    run= runs.iloc[[index]]
    print(f"fetched run {index}")
    return run

def set_job_status(index = None, status = None, db=None):
    assert(status is not None)
    assert(index is not None)
    assert(db is not None)
    experiment_table = sql.Table('experiments', sql.MetaData(db),autoload=True)
    update = experiment_table.update()\
            .where(experiment_table.c.index == index)\
            .values(status = status.value)
    db.execute(update).close()

def fetch_job_status(index = None, db = None):
    assert(index is not None)
    assert(db is not None)
    experiment_table = sql.Table('experiments', sql.MetaData(db),autoload=True)
    s = sql.select([experiment_table.c.status]).where(experiment_table.c.index == index)
    res=db.execute(s)
    return res.fetchone()[0]


def run_parallel(jobid):
    run = fetch_run(int(jobid), db)
    settings = json.loads(run.iloc[0]['settings'])
    set_job_status(int(jobid), JobStatus.IN_PROGRESS, db)
    try:
        from nsga2 import run_algorithm
        run_algorithm(int(jobid), db, **settings)
    except Exception as e:
        print(f"Error running experiment {jobid}: {e}")
        set_job_status(int(jobid), JobStatus.FAILED, db)
        raise e
    set_job_status(int(jobid), JobStatus.DONE, db)
        

import cred
db = sql.create_engine(f'postgresql+psycopg2://{cred.user}:{cred.passwd}@{cred.vm}')
if __name__ == "__main__":
    import os
    
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', default='run')
    parser.add_argument('--index', action='store', default=-1)
    parser.add_argument('--nonslurm', action='store_true')
    args = parser.parse_args()
    poolsize = 27
    if args.nonslurm:
        import multiprocessing as mp
        mp.set_start_method("spawn")
        indexes = list(range(int(args.index),int(args.index)+poolsize))
        with mp.Pool(poolsize) as pool:
            if args.mode == 'run':
                pool.map(run_parallel, indexes)
    else:       
        if int(args.index) >= 0:
            jobid = int(args.index)
        print(f"Starting experiment {jobid}")
        if args.mode == 'create':
            create_experiments()
        if args.mode == 'data':
            create_experiments_from_files()
        if args.mode == 'run':
            from nsga2 import run_algorithm
            print(f"running experiment {jobid}")
            jobstatus = fetch_job_status(int(jobid), db) 
            if jobstatus == JobStatus.TODO or jobstatus==JobStatus.FAILED:
                run = fetch_run(int(jobid), db)
                settings = json.loads(run.iloc[0]['settings'])
                set_job_status(int(jobid), JobStatus.IN_PROGRESS, db)
                try:
                    run_algorithm(int(jobid), db, **settings)
                except Exception as e:
                    print(f"Error running experiment {jobid}: {e}")
                    set_job_status(int(jobid), JobStatus.FAILED, db)
                    raise e
                set_job_status(int(jobid), JobStatus.DONE, db)
            else:
                print(f"Job {jobid} already {JobStatus(jobstatus)}") 


        
