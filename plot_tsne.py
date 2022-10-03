import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
df = pd.read_pickle("./merged_dataset.pkl")

# df = merged_df.copy()
df = df[(df.User == 'sugandh') | (df.User == 'anirban')]
df.loc[df["activity"] == "Talking back", "activity"] = "Turning back"
df.loc[df["activity"] == "Talking Phone", "activity"] = "Using Phone"
df.loc[df["activity"] == "Harsh driving", "activity"] = "Anomaly in Steering"

plotting = {}
for key in list(df.groupby('activity').apply(list).to_dict().keys()):
    if key == 'Normal driving':
        continue
    drowsy_df = df[df.activity == key]
    norm_df = df[df.activity == 'Normal driving'][500:len(drowsy_df) + 500]
    m_df = pd.concat([norm_df, drowsy_df], ignore_index=True)
    plotter = pd.DataFrame()
    col_name = key + 'feat'
    doppz_flat = np.array(m_df['doppz'].values.tolist()).reshape(-1, 16 * 64)
    rp_y = np.array(m_df['rp_y'].values.tolist())
    noiserp_y = np.array(m_df['noiserp_y'].values.tolist())
    rps = np.concatenate((rp_y, noiserp_y), axis=1)
    plotter[col_name] = list(np.concatenate((doppz_flat, rps), axis=1))
    plotter['y'] = m_df['activity'].values

    tsne = TSNE(n_components=2, verbose=0, perplexity=25, n_iter=1600)
    tsne_pca_results = tsne.fit_transform(plotter[col_name].values.tolist())

    plotter['tsne-one'] = tsne_pca_results[:, 0]
    plotter['tsne-two'] = tsne_pca_results[:, 1]
    plotting[key] = plotter

# fig = plt.figure(figsize=(16,8))

fig = plt.figure(figsize=(16, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 40)
j = 1
colors = sns.color_palette("tab10", 11)
col1 = colors[0]

for i, (key, df) in enumerate(plotting.items()):
    if i < 5:
        row_val = 0
    else:
        row_val = 1
        if i >= 5:
            i = (i - 5)
    ax1 = fig.add_subplot(gs[row_val, 8 * i + row_val * 4 + 1:8 * (i + 1) + row_val * 4])
    pos1 = ax1.get_position()
    pos2 = [pos1.x0, pos1.y0, pos1.width / 1.3, pos1.height / 1.3]
    ax1.set_position(pos2)
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="y",
        palette=(col1, colors[j]),
        data=df,
        legend="full",
        alpha=0.3,
        size="y",
        ax=ax1
    )
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=1)
    j += 1
    ax1.grid(alpha=0.4)

fig.tight_layout()
fig.savefig('tsne.svg')
fig.show()
plt.show()
