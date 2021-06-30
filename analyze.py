import sqlalchemy as sql
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json



#for slurm
#db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
#for xdw
db = sql.create_engine('postgresql+psycopg2:///dweikert')


def strip_spaces(s):
    return s.replace(' ','')


results = pd.read_sql('results', con=db)
results.sort_values('index', inplace=True)
#397-462 are with new decision maker


old_results = pd.read_sql('results_old', con=db)
old_results.sort_values('index', inplace=True)
#print(len(results))
#print(list(results))
done = sorted(list(old_results['index']))
#for i in range(199):
#    if not (i in done):
#        try:
#            with open(f'results/{i}.csv') as ifile:
#                df = pd.read_csv(ifile, skipinitialspace=True, converters={'algorithm': strip_spaces})
#                old_results = pd.concat([df, results])
#        except FileNotFoundError as e:
#            pass
print(len(old_results))
old_results.set_index('index', inplace=True)
results.set_index('index', inplace=True)
#results.set_index('index', inplace=True)
#print(results)
pd.set_option('max_colwidth', 1000)
#print(results.iloc[397:463]['settings'])
#old_results.set_index('index', inplace=True)
#print(len(results))
#print(list(results))

def reduce(x):
    if x > 3000:
        x=x/10 
    return x

def get_lat(s):
    lats = json.loads(s)
    return max(lats)


def clean_df(x):
    x = json.loads(x)
    x = max([x1 for x1 in x if int(x1) < 9999]) 
    return x
def getRows(results, old_results, network, task, nNodes, mode = 'olddec'):
    
    #print(json.loads(results.iloc[2]['settings'])['nNodes'])
    if mode =='olddec':
        df = results[results['settings'].str.contains(network)]
        df = df[df['settings'].str.contains(task)]
        df = df[df['settings'].str.contains(f'"nNodes": {nNodes}')]
        if task != 'OneSink':
            df = df.iloc[0:130]
        else:
            df = df.iloc[0:160]
    elif mode == 'newdec':
        #397-462 are with new decision maker
        df = results[results['settings'].str.contains(f'"NGEN_realloc": 10')]
        df = df[df['settings'].str.contains('"algorithm": "rmota"')]
        df2 = results[results['settings'].str.contains('dtas')]
        df2 = df2.iloc[0:33]
        df2 = df2[df2['settings'].str.contains(task)]
        #df2 = df2[df2['settings'].str.contains(f'"NGEN_realloc": 10')]
        df = pd.concat([df,df2])
        df['actual_latency'] = df['actual_latencies'].apply(clean_df)
    elif mode == 'moregen':
        df = results[results['settings'].str.contains(f'"NGEN_realloc": 20')]
    elif mode == 'merge':
        df = results[results['settings'].str.contains(network)]
        df = df[df['settings'].str.contains(task)]
        df = df[df['settings'].str.contains(f'"nNodes": {nNodes}')]
        if task != 'OneSink':
            df = df.iloc[0:130]
        df2 = old_results[old_results['settings'].str.contains(network)]
        df2 = df2[df2['settings'].str.contains(task)]
        df2 = df2[df2['settings'].str.contains(f'"nNodes": {nNodes}')]
        print(len(df2))
        print(len(df))
        df = pd.concat([df,df2])
        df = df[df['actual_latency'] <6000]
    #df = df[df['settings'].str.contains(f'"nTasks": {nTasks}')]
    print(df.columns)
    df['ntasks'] = df['settings'].apply(lambda x : json.loads(x)['nTasks'])
    if mode =='merge':
        df = df[df['ntasks'] >20]
    print(list(df))
    print(f"Before Pruning: {len(df)}")
    #df = df[df['actual_lifetime'] > -100]
    #df['latency_met'] = df['actual_latencies'].apply(get_lat)
    #df['latency_met'] = df['latency_met'].apply(reduce)
    
    #df = df[df['actual_latency'] < 9999]
    print(f"After Pruning: {len(df)}")
    return df

print(list(results))
nw = 'Grid'
task = 'EncodeDecode'
nNodes = '81'
mode = 'merge'
df = getRows(results, old_results, nw, task, nNodes, mode)
#df_old = getRows(old_results, nw, task, nNodes)
#df = pd.concat([df,df_old])
df['algorithm'] = df['algorithm'].str.strip()
def rename(x):
    if x == 'rmota':
        return 'A-MOTA'
    elif x == 'nsga2':
        return 'MOTA'
    elif x == 'dtas':
        return 'DTAS'

df['algorithm'] = df['algorithm'].apply(rename)
print(df['ntasks'])


g = sns.catplot(x='ntasks', y='actual_lifetime', hue='algorithm', data=df, kind='box')
g.set(ylabel='Network Lifetime (s)')
plt.savefig(f"plots/rmota_nl_{mode}_{nw}_{task}_{nNodes}.png")
plt.show()

g = sns.catplot(x='ntasks', y='actual_latency', hue='algorithm', data=df, kind='box')
#g.fig.get_axes()[0].set_yscale('log')
g.set(ylabel='Latency (ms)')
plt.savefig(f"plots/rmota_l_{mode}_{nw}_{task}_{nNodes}.png")
plt.show()
