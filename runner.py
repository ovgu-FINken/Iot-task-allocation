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
    task_numbers = [19,55]
    algorithms = ['rmota','mmota']
    
    network_creator = 'Manhattan'
    task_creator = 'EncodeDecode'
    #task_creator = 'OneSink'
    mobl = [0,30,40]
    for static in [False, True]:
        predl = ['perfect', 'target'] if not static else ['perfect']
        for predictor in predl:
            print(f"creating {predictor} predictor data")
            mobl = [0,30,40]
            mobl = mobl if not static else [0]
            for nMob in mobl:
                mobileNodes = nMob
                nNodes = 81 + mobileNodes
                for i in range(11):
                    with open(f"datasets/{'static' if static else 'mobile'}/{nNodes}/{predictor if not static else ''}/settings_{i}.json") as f:
                        settings = json.load(f)
                        crossover = 'nsga2'
                        NGEN = 50
                        settings.update({'crossover' : crossover})
                        settings.update({'NGEN' : NGEN})
                        for algorithm in algorithms:
                            for nTasks in task_numbers:
                                settings.update({'nTasks' : nTasks})
                                settings.update({'experiment' : 'mmota'})
                                settings.update({'task_creator' : task_creator})
                                settings.update({'algorithm' : algorithm})
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
                    run_algorithm(int(jobid), db, **settings)
                except Exception as e:
                    print(f"Error running experiment {jobid}: {e}")
                    set_job_status(int(jobid), JobStatus.FAILED, db)
                    raise e
                set_job_status(int(jobid), JobStatus.DONE, db)
            else:
                print(f"Job {jobid} already {JobStatus(jobstatus)}") 


        
