import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (16, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

df = pd.read_pickle("./merged_dataset.pkl")

act = ['Normal driving', 'Talking Phone', 'Harsh driving', 'Talking left',
       'Drinking', 'Talking back', 'Nodding', 'Fetching forward', 'Yawning', 'Picking drops']
usr = ['anirban', 'sugandh']

column = 'rp_y'

newdf = pd.DataFrame(columns=['activity', column, 'User'])
for i, u in enumerate(usr):
    for a in act:
        temp = pd.DataFrame(columns=['activity', 'noiserp_y', 'User'])
        df1 = df[df.User == u].copy()
        df2 = df1[df1.activity == a].copy()
        arr1 = np.array(df2[column].apply(lambda e: e.flatten().tolist()).values.tolist()).mean(axis=1)
        mu = arr1.mean()
        sig = arr1.std()
        temp[column] = arr1[np.abs(arr1 - mu) < 2 * sig]
        temp['activity'] = a
        temp['User'] = f'User{i + 1}'
        newdf = pd.concat([newdf, temp], axis=0)


def plotspline(arr, ax, N=100, x=np.arange(0, 10), kim=2, col='r', ls='--'):
    t, c, k = interpolate.splrep(x, arr, s=0, k=kim)
    xx = np.linspace(x.min(), x.max(), N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    ax.scatter(x, arr, c=col)
    ax.plot(xx, spline(xx), c=col, linestyle=ls, linewidth=3)


fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(111)
med = sns.violinplot(x='activity', y=column, hue='User', data=newdf, split=True, ax=ax, scale="count", inner="quartile",
                     order=['Normal driving', 'Yawning', 'Talking Phone', 'Talking left',
                            'Drinking', 'Talking back', 'Picking drops', 'Nodding', 'Fetching forward',
                            'Harsh driving'])
left = [None]
right = [None]
lft = True
for i, l in enumerate(med.lines):
    if i > 0 and i % 3 == 0:
        lft = not lft
    if lft:
        left.append(l.get_data())
    else:
        right.append(l.get_data())

left_med = [e[1][0] for i, e in enumerate(left) if (i + 1) % 3 == 0]
right_med = [e[1][0] for i, e in enumerate(right) if (i + 1) % 3 == 0]

plotspline(left_med, ax, col='#334f7a', ls='-')
plotspline(right_med, ax, col='#915736')

plt.setp(ax.collections, alpha=.7)

ax.set_xticklabels(['Normal\ndriving','Yawning', 'Using\nPhone','Talking\nleft','Drinking',
                    'Turning\nback', 'Picking\ndrops','Nodding',
                    'Fetching\nforward','Anomaly\nin steering'])
plt.xticks(rotation=30)
plt.ylim(76,85)
plt.legend(loc='upper left')
plt.xlabel('')
plt.ylabel('Mean Range Profile')
plt.grid()
plt.show()
