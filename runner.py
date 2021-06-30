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






def create_experiments():
    #task_numbers = [19,43,79]
    task_numbers =[20,40,80]
    algorithms = ['rmota','nsga2','dtas']
    
    network_creator = 'Grid'
    #task_creator = 'EncodeDecode'
    task_creator = 'OneSink'
    nNodes = 81
    dims = 9
    energy = 100
    energy_list = [energy]*nNodes
    network_status = [1]*nNodes
    crossover = 'nsga2'
    NGEN = 100
    for algorithm in algorithms:
        for nTasks in task_numbers:
            for i in range(11):
                settings = {
                         'experiment' : 'rmota',
                         'nNodes' : nNodes,
                         'network_creator' : network_creator,
                         'dimx' : dims,
                         'dimy' : dims,
                         'nTasks' : nTasks,
                         'task_creator' : task_creator,
                         'energy_list_sim' : energy_list,
                         'energy_list_eval': energy_list,
                         'init_energy' : energy,
                         'algorithm' : algorithm,
                         'crossover' : crossover,
                         'verbose' : False,
                         'capture_packets' : False,
                         'enable_errors' : False,
                         'error_shape' : 1.0,
                         'error_scale' : 1.0,
                         'network_status' : network_status,
                         'NGEN' : NGEN,
                         'NGEN_realloc': 20,
                         'run_number' : 1,
                         'experiment_number' : i,
                         'seed' : 2002+i*42,
                         'next_alloc' : [],
                         }
                old_results = pd.read_sql("experiments", con=db)
                min_index = old_results.index.max() + 1 if len(old_results) > 0 else 0
                run = {'index' : min_index,
                   'experiment' : 'rmota',
                   'algorithm' : algorithm,
                   'status' : JobStatus.TODO,
                   'settings' : json.dumps(settings)
                    }
                df = pd.DataFrame(run, index=[0])
                df.set_index('index', inplace=True)
                df.to_sql('experiments', db, if_exists='append')
                

def fetch_run(index, db):
    runs = pd.read_sql("experiments", con=db)
    runs = runs.sort_values(by='index')
    print(runs)
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
        

db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
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
        if args.mode == 'run':
            from nsga2 import run_algorithm
            print(f"running experiment {jobid}")
            jobstatus = fetch_job_status(int(jobid), db) 
            if jobstatus == JobStatus.TODO or jobstatus==JobStatus.FAILED:
                run = fetch_run(int(jobid), db)
                settings = json.loads(run.iloc[0]['settings'])
                set_job_status(int(jobid), JobStatus.IN_PROGRESS, db)
                try:
                    run_algorithm(int(jobid),db, **settings)
                except Exception as e:
                    print(f"Error running experiment {jobid}: {e}")
                    set_job_status(int(jobid), JobStatus.FAILED, db)
                    raise e
                set_job_status(int(jobid), JobStatus.DONE, db)
            else:
                print(f"Job {jobid} already {JobStatus(jobstatus)}") 


        
