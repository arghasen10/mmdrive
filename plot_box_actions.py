import pandas as pd
from datetime import datetime
import json
import numpy as np
import glob
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})
# plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def process_mmwave(f):
    user, datetime_str = f.split('/')[-1].split('-')
    datetime_str = datetime_str.split('.')[0].split('_')[0]
    datetime_str = datetime.strftime(datetime.strptime(datetime_str, "%Y%m%d"), "%Y-%m-%d") + ' '
    try:
        data = [json.loads(val) for val in open(f, "r")]
    except Exception as e:
        data = []
        for val in open(f, "r"):
            try:
                data.append(json.loads(val))
            except Exception as e:
                print(e)
                continue
    mmwave_df = pd.DataFrame()
    for d in data:
        mmwave_df = mmwave_df.append(d['answer'], ignore_index=True)

    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: datetime_str + ':'.join(e.split('_')))
    mmwave_df['User'] = user
    mmwave_df['rp_y'] = list(np.array(mmwave_df['rp_y'].values.tolist()))
    mmwave_df['noiserp_y'] = list(np.array(mmwave_df['noiserp_y'].values.tolist()))
    mmwave_df['doppz'] = list(np.array(mmwave_df['doppz'].values.tolist()))
    mmwave_df = mmwave_df[['datetime', 'rp_y', 'noiserp_y', 'doppz', 'User']]
    return mmwave_df


def read_activity():
    files = glob.glob('driving**.csv')

    activity_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    activity_df = activity_df[activity_df.activity != 'Random']
    return activity_df


def read_mmwave():
    files = glob.glob('driving_dataset/sugandh*.txt')

    mmwave_df = pd.concat([process_mmwave(f) for f in files], ignore_index=True)
    print(mmwave_df.head())
    print(mmwave_df.values.shape)
    return mmwave_df


activity_df = read_activity()
mmwave_df = read_mmwave()

merged_df = mmwave_df.merge(activity_df, left_on='datetime', right_on='datetime', how='inner')

box_plotter = {}
noise_plotter = {}
df = merged_df[merged_df.User == 'sugandh'].copy()
for key in list(df.groupby('activity').apply(list).to_dict().keys()):
    box_plotter[key] = []
    noise_plotter[key] = []
for d in range(len(df)):
    box_plotter[df.iloc[d:d + 1]['activity'].values[0]].append(df.iloc[d:d + 1]['doppz'].values[0].mean())
    noise_plotter[df.iloc[d:d + 1]['activity'].values[0]].append(df.iloc[d:d + 1]['noiserp_y'].values[0].mean())

ax = plt.subplot()
ax.boxplot([box_plotter['Normal driving'], box_plotter['Drinking'], box_plotter['Fetching forward'],
            box_plotter['Harsh driving'], box_plotter['Nodding'], box_plotter['Picking drops'],
            box_plotter['Talking Phone'], box_plotter['Talking back'], box_plotter['Talking left'],
            box_plotter['Yawning']], patch_artist=True, notch=True, positions=np.arange(1, 11) - 0.2, widths=0.4,
           showfliers=False)
ax.set_ylabel('Mean range-doppler')
ax2 = ax.twinx()
ax2.set_ylabel('Mean noise profile')
c = 'tab:green'
ax2.boxplot([noise_plotter['Normal driving'], noise_plotter['Drinking'], noise_plotter['Fetching forward'],
             noise_plotter['Harsh driving'], noise_plotter['Nodding'], noise_plotter['Picking drops'],
             noise_plotter['Talking Phone'], noise_plotter['Talking back'], noise_plotter['Talking left'],
             noise_plotter['Yawning']], patch_artist=True, notch=True, positions=np.arange(1, 11) + 0.2, widths=0.4,
            boxprops=dict(facecolor=c, color=c), capprops=dict(color=c), showfliers=False
            )
ax.set_xticks(range(1, 11))
ax.set_xticklabels(['Normal\ndriving', 'Drinking', 'Fetching\nforward', 'Anomaly\nin\nsteering', 'Nodding',
                    'Picking\ndrops', 'Using\nPhone', 'Turning\nback', 'Talking\nleft', 'Yawning'])
# plt.grid(alpha=0.2)
plt.show()
