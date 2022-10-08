import pandas as pd
from datetime import datetime
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 12})
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
    files = glob.glob('driving_dataset/*.txt')

    mmwave_df = pd.concat([process_mmwave(f) for f in files], ignore_index=True)
    print(mmwave_df.head())
    print(mmwave_df.values.shape)
    return mmwave_df


activity_df = read_activity()
mmwave_df = read_mmwave()

merged_df = mmwave_df.merge(activity_df, left_on='datetime', right_on='datetime', how='inner')
# merged_df = pd.read_pickle('merged_dataset.pkl')
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

df = merged_df[merged_df.User == 'sugandh']
df = df[(df.datetime < '2022-09-25 14:59:46') & (df.datetime > '2022-09-25 14:21:19')]
df1 = df.iloc[900:2000]
df1.activity.reset_index()
doppz = np.array(df1['doppz'].values.tolist())
fig = plt.figure(figsize=(15, 6))

ax = fig.add_subplot(221)

map_dict = dict(zip(['Fetching forward', 'Harsh driving', 'Normal driving', 'Picking drops', 'Talking Phone',
                     'Talking back', 'Talking left', 'Yawning'],
                    ['orange', 'blue', 'red', 'green', 'indigo', 'gold', 'maroon', 'skyblue']))
labels = df1['activity'].map(map_dict)
sum_val = doppz.reshape(-1, 64 * 16).mean(axis=1)
sum_arr = sum_val
ax.scatter(range(len(sum_arr)), sum_arr, c=labels)
ax.plot(range(len(sum_arr)), sum_arr, linestyle='--', c='azure')
ax.set_ylabel('Mean Range-Doppler')
ax.set_xlim(0, 700)
ax.set_ylim(2180, 2600)
xticks = np.linspace(0, 700, 8, dtype=np.int)
xtickslabel = np.linspace(0, 140, 8, dtype=np.int)
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabel)
ax.set_xlabel('Time(s)')
plt.grid()

pathes = [mpatches.Patch(color=c, label=v) for v, c in map_dict.items()]
ax.legend(handles=pathes, ncol=5, bbox_to_anchor=(2, 1.35))
ax.set_xlim(0, 700)
ax = fig.add_subplot(222)
noiserp_y = np.array(df1['noiserp_y'].values.tolist())
sum_val = noiserp_y.mean(axis=1)
sum_arr = sum_val
ax.scatter(range(len(sum_arr)), sum_arr, c=labels)
ax.plot(range(len(sum_arr)), sum_arr, linestyle='--', c='azure')
ax.set_ylabel('Mean Noise Profile')
ax.set_xlim(0, 700)
ax.set_ylim(30, 47)
xticks = np.linspace(0, 700, 8, dtype=np.int)
xtickslabel = np.linspace(0, 140, 8, dtype=np.int)
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabel)
ax.set_xlabel('Time(s)')
plt.grid()

df1 = df.iloc[2800:3800]
doppz = np.array(df1['doppz'].values.tolist())
labels = df1['activity'].map(map_dict)
sum_val = doppz.reshape(-1, 64 * 16).mean(axis=1)
ax = fig.add_subplot(223)
sum_arr = sum_val
ax.scatter(range(len(sum_arr)), sum_arr, c=labels)
ax.plot(range(len(sum_arr)), sum_arr, linestyle='--', c='azure')
ax.set_ylabel('Mean Range-Doppler')
ax.set_xlim(0, 700)
ax.set_ylim(2180, 2600)
xticks = np.linspace(0, 700, 8, dtype=np.int)
xtickslabel = np.linspace(200, 340, 8, dtype=np.int)
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabel)
ax.set_xlabel('Time(s)')
plt.grid()

ax = fig.add_subplot(224)
noiserp_y = np.array(df1['noiserp_y'].values.tolist())
labels = df1['activity'].map(map_dict)
sum_val = noiserp_y.mean(axis=1)
sum_arr = sum_val
ax.scatter(range(len(sum_arr)), sum_arr, c=labels)
ax.plot(range(len(sum_arr)), sum_arr, linestyle='--', c='azure')
ax.set_ylabel('Mean Noise Profile')
ax.set_xlim(0, 700)
ax.set_ylim(30, 47)
xticks = np.linspace(0, 700, 8, dtype=np.int)
xtickslabel = np.linspace(200, 340, 8, dtype=np.int)
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabel)
ax.set_xlabel('Time(s)')
plt.grid()
plt.show()
