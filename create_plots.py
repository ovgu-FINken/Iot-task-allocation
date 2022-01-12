import pandas as pd
import sqlalchemy as sql
import matplotlib.pyplot as plt
import os
import seaborn as sns


db = sql.create_engine('postgresql+psycopg2://dweikert:mydbcuzwhohacksthis@10.61.14.160:5432/dweikert')
df =  pd.read_sql('results_final_mobility', db)



#df = df[df['static']==False]
#df2 = df[df['predictor'] == 'perfect']
#df3 = df2[df2['st
#df = pd.concat([df2,df3])

df2 = df[df['predictor']=='perfect']
df2 = df2[df2['algorithm'] =='mmota']
df2 = df2.assign(algorithm= 'M-MOTA_perfect')
print(len(df2))


df4 = df[df['predictor']=='perfect']
df4 = df4[df4['algorithm'] =='rmota']

df3 = df[df['predictor']=='target']

df = pd.concat([df3,df2,df4])
df['algorithm'].replace({'rmota' : "A-MOTA"}, inplace=True)
df['algorithm'].replace({'mmota' : "M-MOTA_target"}, inplace=True)

sns.set_style("whitegrid")
g = sns.catplot(x = 'nnodes', y = 'lifetime', hue='algorithm', data=df, kind='box')
g._legend.remove()
plt.xlabel('Number of nodes')
plt.ylabel('- Network Lifetime (s)')
plt.savefig("plots/nl.png")
plt.show()

g = sns.catplot(x = 'nnodes', y = 'latency', hue='algorithm', data=df, kind='box')
plt.xlabel('Number of nodes')
plt.ylabel('Latency (ms)')
plt.savefig("plots/l.png")
plt.show()

g = sns.catplot(x = 'nnodes', y = 'nMissed', hue='algorithm', data=df, kind='box')
plt.xlabel('Number of nodes')
plt.ylabel('Missed packets')
plt.savefig("plots/missed.png")
plt.show()
