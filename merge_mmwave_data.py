import pandas as pd
import glob
from datetime import datetime
import json


def process_mmwave(f):
    user, datetime_str = f.split('-')
    datetime_str = datetime_str.split('.')[0].split('_')[0]
    datetime_str = datetime.strftime(datetime.strptime(datetime_str, "%Y%m%d"), "%Y-%m-%d")+' '
    data = [json.loads(val) for val in open(f, "r")]
    mmwave_df = pd.DataFrame()
    for d in data:
        mmwave_df = mmwave_df.append(d['answer'], ignore_index=True)

    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: datetime_str + ':'.join(e.split('_')))
    mmwave_df = mmwave_df[['datetime', 'rp_y', 'noiserp_y', 'doppz']]
    return mmwave_df


files = glob.glob('activ**.csv')

activity_df = pd.concat([pd.read_csv(f) for f in files])
activity_df = activity_df[activity_df.activity != 'Eating']

files = glob.glob('*.txt')
dfs = []


mmwave_df = pd.concat([process_mmwave(f) for f in files])
print(mmwave_df.head())
print(mmwave_df.values.shape)