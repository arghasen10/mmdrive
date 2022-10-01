import seaborn as sns
import pandas as pd
from datetime import datetime
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# plt.style.use('ggplot')

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")


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

df = merged_df[merged_df.User == 'sugandh']
doppz = np.array(df['doppz'].values.tolist())

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

map_dict = dict(zip(['Drinking', 'Fetching forward', 'Harsh driving', 'Nodding', 'Normal driving', 'Picking drops',
                     'Talking Phone', 'Talking back', 'Talking left', 'Yawning'],
                    ['k', 'orange', 'blue', 'cyan', 'red', 'green', 'indigo', 'gold', 'maroon', 'skyblue']))
labels = df['activity'].map(map_dict)
sum_val = doppz.reshape(-1, 64 * 16).mean(axis=1)
ax0.scatter(range(len(sum_val)), sum_val, c=labels)
ax0.plot(range(len(sum_val)), sum_val, linestyle='--', c='azure')
ax0.set_ylabel('Average Range-Doppler Variation')

pathes = [mpatches.Patch(color=c, label=v) for v, c in map_dict.items()]
ax0.legend(handles=pathes, ncol=3, bbox_to_anchor=(0.5, 1.05))

rpy = np.array(df['rp_y'].values.tolist())
sum_val = rpy.mean(axis=1)
ax1.scatter(range(len(sum_val)), sum_val, c=labels)
ax1.plot(range(len(sum_val)), sum_val, linestyle='--', c='azure')
ax1.set_ylabel('Average Range Profile Variation')

noiserp_y = np.array(df['noiserp_y'].values.tolist())
sum_val = noiserp_y.mean(axis=1)
ax2.scatter(range(len(sum_val)), sum_val, c=labels)
ax2.plot(range(len(sum_val)), sum_val, linestyle='--', c='azure')
ax2.set_ylabel('Average Noise Profile Variation')

noiserp_y = np.subtract(rpy, noiserp_y)
sum_val = noiserp_y.mean(axis=1)
ax3.scatter(range(len(sum_val)), sum_val, c=labels)
ax3.plot(range(len(sum_val)), sum_val, linestyle='--', c='azure')
ax3.set_ylabel('Average SNR Profile Variation')
plt.tight_layout()
plt.grid(alpha=0.2)
plt.show()

fig = plt.Figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 40)

doppz = np.array(df[164:180]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
print('plotting Drinking')
ax1 = fig.add_subplot(gs[0, 0:8])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax1, vmax=0.28, vmin=0)
g1.set_xlabel('Drinking')
# plt.show()

print('plotting Talking Phone')
doppz = np.array(df.iloc[270:285]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax2 = fig.add_subplot(gs[0, 8:16])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax2, vmax=0.28, vmin=0)
g1.set_xlabel('Using Phone')
# plt.show()

doppz = np.array(df.iloc[762:781]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax3 = fig.add_subplot(gs[0, 16:24])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax3, vmax=0.28, vmin=0)
g1.set_xlabel('Anomaly in Steering')

doppz = np.array(df.iloc[1495:1510]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax4 = fig.add_subplot(gs[0, 24:32])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax4, vmax=0.28, vmin=0)
g1.set_xlabel('Turning back')

doppz = np.array(df.iloc[2055:2070]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax5 = fig.add_subplot(gs[0, 32:40])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax5, vmax=0.28, vmin=0)
g1.set_xlabel('Picking Drops')

doppz = np.array(df.iloc[2365:2380]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax6 = fig.add_subplot(gs[1, 4:12])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax6, vmax=0.28, vmin=0)
g1.set_xlabel('Fetching forward')
doppz = np.array(df.iloc[2690:2700]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax7 = fig.add_subplot(gs[1, 12:20])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax7, vmax=0.28, vmin=0)
g1.set_xlabel('Talking left')
doppz = np.array(df.iloc[3170:3215]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax8 = fig.add_subplot(gs[1, 20:28])
g1 = sns.heatmap(doppz.std(axis=0), cbar=False, ax=ax8, vmax=0.28, vmin=0)
g1.set_xlabel('Nodding')
doppz = np.array(df.iloc[3703:3718]['doppz'].values.tolist())
doppz = doppz[:, :, :32]
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
ax9 = fig.add_subplot(gs[1, 28:36])
axcb = fig.add_subplot(gs[1, 36:37])
g1 = sns.heatmap(doppz.std(axis=0), ax=ax9, cbar_ax=axcb, cbar_kws={"shrink": 0.5}, vmax=0.3)
g1.set_xlabel('Yawning')
g1.set_xticks([0, 16, 32])
g1.set_yticks([0, 8, 16])
fig.savefig('range_doppler_variation.eps')
fig.show()
