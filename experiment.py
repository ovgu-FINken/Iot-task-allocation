import git 
import json
import os.time



from nsga2 import nsga2


class JobStatus(IntEnum):
    TODO = 0 
    IN_PROGRESS = 1 
    DONE = 2 
    FAILED = 3

algorithms = {  'nsga2' : nsga2 }

def get_commit()
    repo = git.Repo(search_parent_directories = True)
    return repo.head.object.hexsha

class Experiment():
    def __init__(self, settings):
        self.settings = settings
    
    def setup(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutRandomNode, networkGraph = networkGraph, taskGraph= taskGraph, indpb=1.0/NDIM)
        toolbox.register("select", tools.selNSGA2)


    def run(self):
        if self.algorithm == find_best_solution:
            return self.algorithm(self.network, self.task_list, **self.settings)
        if self.algorithm == genetic_task_allocation:
            return self.algorithm(self.network, self.task_list, self.pop, self.toolbox, **self.settings)
        return self.algorithm(self.network, self.task_list, **self.settings)

class ExperimentRunner():
    def __init__(self,db):
        assert db.has_table("jobs")
        self.db = db 
        self.metadata = sqlalchemy.MetaData(self.db)
        self.table_jobs = sqlalchemy.Table('jobs', self.metadata, autoload = True)

    def fetch_job(self, verbose = False):
        select = sqlalchemy.sql.select([self.table_jobs]).where( (self.table_jobs.c.status == JobStatus.TODO.value) | (self.table_jobs.c.status == JobStatus.FAILED.value))
        r = self.db.execute(select)
        row = r.fetchone()
        r.close()
        if row is None:
            if verbose:
                print("could not fetch job")
            return
        if row[self.table_jobs.c.commit] != get_commit():
            if verbose:
                print("WARNING: commits do not match")
        job = {}
        for col in self.table_jobs.columns.keys():
            job[str(col)] = row[col]
        job['settings'] = json.loads(job['settings'])
        if verbose:
            print(f"fetched job: {job}")
        return job 

    def set_job_status(self, job=None, status=None, time=0):
        assert status is not None 
        if job is None:
            print(f"Tying to set job None to {status}")
            return 
        update = self.table_jobs.update()\
                .where(self.table_jobs.c.index == job['index'])\
                .values(status=status.value, pid = os.getpid(), time=int(time))
        self.db.execute(update).close()

    def execute_and_save(self,job):
        print(f"executing job {job['index']}")
        self.set_job_status(job, status=JobStatus.IN_PROGRESS)
        try:
            res = execute_job(job)
            save_results(res, job)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            self.set_job_status(job, status = JobStatus.FAILED)
            raise 
        return True
    
    def execute_pool(self, workers=31):
        with Pool(processes=workers) as pool:
            jobs={}
            handles={}
            try:
                while True:
                    job = self.fetch_job()
                    if job is not None and len(handles.keys()) < workers:
                        self.set_job_status(job, status = JobStatus.IN_PROGRESS)
                        jobs[job['index']] = job
                        handles[job['index']] = pool.apply_async(execute_job, (job,))
                        print(f"{time.strftime('%H:%M:%S')} -- start {job['index']}.")
                    if len(handles.keys()) == workers:
                        time.sleep(5)
                    completed = []
                    for k, v in handles.items():
                        if v.ready():
                            completed.append(k)
                            if v.successful():
                                self.save_results(v.get(), jobs[k])
                                print(f"{time.strftime('%H:%M:%S')} -- job {k} successful")
                            else:
                                self.set_job_status(jobs[k], status=JobStatus.FAILED)
                                print(f"{time.strftime('%H:%M:%S')} -- job {k} failed")
                                try:
                                    print(v.get())
                                except:
                                    pass 
                    for k in completed:
                        del handles[k]
                        del jobs[k]
                    time.sleep(0.5)
            except Exception as e:
                print(e)
                for k,v in jobs.items():
                    self.set_job_status(v, status=JobStatus.FAILED)
                raise 
            return True

    def save_results(self, res, job):
        lifetime = res['stats'][-1]['lifetime']
        latency = res['stats'][-1]['latency']
        self.save_lifetime(lifetime, job = job)
        self.save_latency(latency, job = job)
        self.save_assignment_data(results = res, lifetime = lifetime, latency = latency, job = job)
        self.set_job_status(job, status=JobStatus.DONE, time=res['job_time'])

    def save_lifetime(self, lifetime=None, job=None):
        if lifetime is None:
            print("Not saving lifetime of None")
            return
        if job is None:
            print("Not saving lifetime for job None")
            return
        df = pd.DataFrame({ 'lifetime' : [lifetime],
                            'job_index' : job['index'],
                            'experiment' : job['experiment']
                            })
        df.set_index('job_index', inplace=True)
        df['run'] = job['run']
        df['seed'] = job['seed']
        df.to_sql("lifetimes", self.db, if_exists = "append")
        print("savef lifetime")

    def save_latency(self, latency = None, job= None):
        if latency is None:
            print("Not saving latency of None")
            return
        if job is None:
            print("Not saving latency for job None")
            return
        df = pd.DataFrame({ 'latency' : [latency],
                            'job_index' : job['index'],
                            'experiment' : job['experiment']
                            })
        df.set_index('job_index', inplace=True)
        df['run'] = job['run']
        df['seed'] = job['seed']
        df.to_sql("latencies", self.db, if_exists = "append")
    print("saved latency")

    def save_assignment_data(self, results = None, lifetime = None, latency = None, job= None):
        if results is None:
            print("Not saving latency of None")
            return    
        if job is None:
            print("Not saving latency for job None")
            return
        df = pd.DataFrame({ 'node_data' : json.dumps(results['node_data']),
                            'assignment_trace' : json.dumps(results['assignment_trace']),
                            'latency' : [latency],
                            'lifetime' : [lifetime],
                            'job_index' : job['index'],
                            'experiment' : job['experiment']
                            })
        df.set_index('job_index', inplace=True)
        df['run'] = job['run']
        df['seed'] = job['seed']
        df.to_sql("run_data", self.db, if_exists = "append")
        print("saved assignment")

def execute_job(job=None):
    assert job is not None 
    start_time = time.time()
    experiment = Experiment(job['settings'])
    try:
        experiment.setup()
    except Exception as e:
        print(f"error in experiment setup: {e}")
        traceback.print_tb(e.__traceback__)
        raise 
    result_dict = experiment.run()
    job_time = time.time() - start_time 
    result_dict.update({'job_time' : job_time})
    return result_dict



def add_jobs_to_db(settings, db = None, experiment = None, runs = 31, delete = False, seed_offset = 1000, time = -1, pid = -1):
    assert experiment is not None
    assert db is not None 
    jobs = [{
        "run" : i,
        "seed": i+seed_offset,
        "status": JobStatus.TODO.value,
        "commit": get_commit(),
        'experiment' : experiment,
        'time' : time,
        'pid' : pid,
        "settings": json.dumps(settings)
        } for i in range(runs)]     
    df_jobs = pd.DataFrame(jobs)
    if delete:
        sql = 'DROP TABLE IF EXISTS lifetimes;'
        db.execute(sql)
        sql = 'DROP TABLE IF EXISTS latencies;'
        db.execute(sql)
        sql = 'DROP TABLE IF EXISTS run_data;'
        db.execute(sql)
        df_jobs.to_sql("jobs", con=db, if_exists="replace")
    else:
        old_jobs = pd.read_sql("jobs", con=db)
        min_index = old_jobs.index.max() + 1 if len(old_jobs) > 0 else 0
        df_jobs.index = range(min_index, min_index + len(df_jobs))
        df_jobs.to_sql("jobs", con=db, if_exists="append")




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', dest ='objectives', action='store', default ='all', choices=('all','first','second'))
    parser.add_argument('--alg', action='store', dest='alg', default = 'genetic', choices = ('genetic', 'pso', 'exhaustive'))
    parser.add_argument('--norunner', action = 'store_false', dest='runner')
    args = parser.parse_args()
    
    """"
    settings = {'task_count_list' : [2,3],
                'link_count_list' : [1,2],
                'graph_count' : 2,
                'node_count' : 15,
                'min_nodes_per_task' : 2,
                'algorithm' : alg,
                'seed' : 424242,
                'objectives' : args.objectives,
                'popsize' : 50,
                'offline_it' : 500,
                'update_interval' : 50}
    settings = settings.get_settings('large1', args.alg, args.objectives)
    #run experiments
    if not args.alg == 'exhaustive'
        engine = sqlalchemy.create_engine('postgresql:///dweikert')
        #runner = ExperimentRunner(engine)
        add_jobs_to_db(settings, engine, experiment = 'exhaustive')
        runner = ExperimentRunner(engine)
        runner.execute_pool(workers = 60)
    else:
        exp = Experiment(settings)
        exp.setup()
        result_list = exp.run()
        with open(f"./results_exhaustive/" + '2312424242' +".pck", 'wb') as ofile:
            pck.dump(result_list, ofile)  
        print(sorted(result_dict, key=lambda x : x[0][0], reverse = True))
    """
    if args.runner:
        if args.alg =='exhaustive':
            exp_name = 'exhaustive1'
            settings = get_settings(name = exp_name, algorithm = args.alg, objectives = args.objectives)
            exp = Experiment(settings)
            exp.setup()
            result_list = exp.run()
            with open(f"./results_exhaustive/{exp_name}.pck", 'wb') as ofile:
                pck.dump(result_list, ofile)  
            #print(sorted(result_list, key=lambda x : x[0][0], reverse = True))
        else:
            exp_name = 'line'
            alg = 'genetic'
            obj = 'all'
            settings = get_settings(name = exp_name, algorithm = alg, objectives = obj)
            engine = sqlalchemy.create_engine('postgresql:///dweikert')
            runner = ExperimentRunner(engine)
            #runner = ExperimentRunner(engine)
            add_jobs_to_db(settings, engine, experiment = exp_name, delete = True)
            exp_name = 'large1'
            settings = get_settings(name = exp_name, algorithm = alg, objectives = obj)
            add_jobs_to_db(settings, engine, experiment = exp_name)
            settings = get_settings(name = exp_name, algorithm = alg, objectives = 'first')
            add_jobs_to_db(settings, engine, experiment = exp_name) 
            runner = ExperimentRunner(engine)
            runner.execute_pool(workers = 60)
        
    else:
        alg = 'pso'
        settings = get_settings(algorithm = 'pso')
        exp = Experiment(settings)
        exp.setup()
        ret = exp.run()
        print(ret)
:
