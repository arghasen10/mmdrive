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

from datetime import datetime, timedelta
import pandas as pd
import glob
import time
import csv

files = glob.glob('/home/argha/Desktop/dangerous_driving_a/**/**/nexar_data/*B.dat')
files.sort()
plot_imu = []
time_then = []
for f in files:
    with open(f, 'r') as file:
        print(f)
        lines = file.readlines()
        for l in lines:
            data = l.strip().split(',')
            if len(data) == 9:
                time_val = int(data[0].split('|')[0]) / 1000000
                actual_time = datetime.strptime(time.strftime('%Y%m%d%H%M%S', time.localtime(time_val)), '%Y%m%d%H%M%S') + \
                              timedelta(days=14, seconds=-3)
#                 if pd.to_datetime('2022-09-25 14:23:53') < actual_time < pd.to_datetime('2022-09-25 14:28:22'):
                time_then.append(actual_time)
                plot_imu.append(float(data[4]))

df_imu=pd.DataFrame({'datetime':map(str,time_then),'imu':plot_imu}).groupby('datetime').mean()

df_user=merged_df[merged_df.User == 'sugandh']
df_user2=\
pd.DataFrame(data=[[str(g),np.array(data['noiserp_y'].values.tolist()).mean(),np.array(data['doppz'].values.tolist()).mean(),
                   data['activity'].values[0]]\
for g, data in df_user.groupby('datetime')],columns=['datetime','noiserp_y','doppz','act'])

df_final = pd.merge(left=df_imu,right=df_user2,on='datetime')
map_dict = dict(zip(['Fetching forward', 'Harsh driving', 'Normal driving', 'Picking drops', 'Talking Phone',
                     'Talking back', 'Talking left', 'Yawning', 'Drinking', 'Nodding'],
                    ['orange', 'blue', 'red', 'green', 'indigo', 'gold', 'maroon', 'skyblue', 'yellow', 'purple']))
df_final['act']=df_final['act'].map(map_dict)

fig = plt.figure(figsize=(12,5))
ax=fig.add_subplot(311)
ax.plot(df_final['imu'].iloc[12:], lw=2)
ax.vlines(x=331,ymin=-3,ymax=3,color='k',linestyle='--',lw=2)
ax.annotate("Bump",(332,-2), (342,-2), arrowprops=dict(facecolor='black', width=1))
ax.set_xlim(207,407)
ax.set_ylim(-3,3)
ax.set_ylabel('IMU')
plt.xticks(np.arange(207, 408,20),labels=map(str,range(0,201,20)))
ax.spines['bottom'].set_visible(False)
ax.set_xticklabels([])
ax.grid()


ax=fig.add_subplot(312)
ax.scatter(range(df_final.shape[0]),df_final['noiserp_y'],c=df_final['act'].values)
ax.plot(range(len(df_final['noiserp_y'])), df_final['noiserp_y'], linestyle='--', c='gray')
ax.vlines(x=324,ymin=32.5,ymax=35.5, color='k',linestyle='--',lw=2)
ax.set_xlim(200,400)
ax.set_ylim(32.5,35.5)
ax.set_ylabel('Mean\nNoise Profile')
plt.xticks(np.arange(200, 401,20),labels=map(str,range(0,201,20)))
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticklabels([])
ax.grid()
pathes = [mpatches.Patch(color=c, label=v) for v, c in map_dict.items()]
ax.legend(handles=pathes, ncol=5, bbox_to_anchor=(1., 3))

ax=fig.add_subplot(313)
ax.scatter(range(df_final.shape[0]),df_final['doppz'],c=df_final['act'].values)
ax.plot(range(len(df_final['doppz'])), df_final['doppz'], linestyle='--', c='gray')
ax.vlines(x=324,ymin=2200,ymax=2350, color='k',linestyle='--',lw=2)

ax.set_xlim(200,400)
ax.set_ylim(2200,2350)
ax.set_ylabel('Mean\nRange-doppler')
plt.xticks(np.arange(200, 401,20),labels=map(str,range(0,201,20)))
ax.spines['top'].set_visible(False)
plt.tight_layout()
ax.set_xlabel('Time(s)')
ax.grid()
plt.show()

fig = plt.figure(figsize=(12,5))
ax=fig.add_subplot(311)
ax.plot(df_final['imu'].iloc[12:], lw=2)
ax.vlines(x=742,ymin=-3,ymax=3,color='k',linestyle='--',lw=2)
ax.annotate("Bump",(743,-2), xytext=(750,-2.1), arrowprops=dict(facecolor='black', width=2))
ax.set_xlim(572,753)
ax.set_ylim(-3,3)
ax.set_ylabel('IMU')
plt.xticks(np.arange(572,753,20),labels=map(str,range(0,181,20)))
ax.spines['bottom'].set_visible(False)
ax.set_xticklabels([])
ax.grid()

ax=fig.add_subplot(312)
ax.scatter(range(df_final.shape[0]),df_final['noiserp_y'],c=df_final['act'].values)
ax.plot(range(len(df_final['noiserp_y'])), df_final['noiserp_y'], linestyle='--', c='gray')
ax.vlines(x=720,ymin=32.5,ymax=38, color='k',linestyle='--',lw=2)
ax.set_xlim(550,731)
ax.set_ylim(32.5,38)
ax.set_ylabel('Mean\nNoise Profile')
plt.xticks(np.arange(550, 731,20),labels=map(str,range(0,181,20)))
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticklabels([])
pathes = [mpatches.Patch(color=c, label=v) for v, c in map_dict.items()]
ax.legend(handles=pathes, ncol=5, bbox_to_anchor=(1., 3))
ax.grid()

ax=fig.add_subplot(313)
ax.scatter(range(df_final.shape[0]),df_final['doppz'],c=df_final['act'].values)
ax.plot(range(len(df_final['doppz'])), df_final['doppz'], linestyle='--', c='gray')
ax.vlines(x=720,ymin=2200,ymax=2450, color='k',linestyle='--',lw=2)

ax.set_xlim(550,731)
ax.set_ylim(2200,2450)
ax.set_ylabel('Mean\nRange-doppler')
plt.xticks(np.arange(550, 731,20),labels=map(str,range(0,181,20)))
ax.spines['top'].set_visible(False)
plt.tight_layout()
ax.set_xlabel('Time(s)')
ax.grid()
plt.show()
