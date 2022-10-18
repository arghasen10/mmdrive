import pandas as pd
import glob
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import det_curve
import matplotlib.patches as mpatches

merged_df = pd.read_pickle('./amp_phase.pkl')
merged_df.loc[merged_df["activity"] == "Harsh Driving", "activity"] = "Anomaly in Steering"
merged_df.loc[merged_df["activity"] == "Yawn", "activity"] = "Yawning"
merged_df.loc[merged_df["activity"] == "Picking Drops", "activity"] = "Picking drops"
merged_df.loc[merged_df["activity"] == "Phone", "activity"] = "Using Phone"
merged_df.loc[merged_df["activity"] == "Turning Back", "activity"] = "Turning back"
merged_df.loc[merged_df["activity"] == "Fetching Forward", "activity"] = "Fetching forward"
merged_df.loc[merged_df["activity"] == "Talking Left", "activity"] = "Talking left"


# ## Preprocessing

def convert_to_neumeric(label):
    lbl_map = {'Drinking': 0,
               'Fetching forward': 1,
               'Anomaly in Steering': 2,
               'Nodding': 3,
               'Yawning': 4,
               'Picking drops': 5,
               'Using Phone': 6,
               'Turning back': 7,
               'Talking left': 8,
               'Normal driving': 9}
    return np.array(list(map(lambda e: lbl_map[e], label)))


def split_dataset(data, label):
    np.random.seed(101)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test


def StackFrames(data, labels, frame_stack=4):
    max_index = data.shape[0] - frame_stack
    stacked_data = np.array([data[i:i + frame_stack] for i in range(max_index)])
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_data, new_labels


class rf_model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = self.rf_process(X_train, X_test, y_train, y_test)
        self.model = self.get_rf_model()

    def PoolOp(self, img, ksize=(16, 16), op=np.mean):
        i_h, i_w, i_c = img.shape
        k_h, k_w = ksize
        row = []
        for c in range(i_c):
            for i in range(i_h // k_h):
                for j in range(i_w // k_w):
                    row.append(op(img[k_h * i:k_h * i + k_h, k_w * j:k_w * j + k_w, c]))
        return np.array(row)

    def subapply(self, v, op):
        return np.concatenate([
            self.PoolOp(np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[0]], axis=2),
                        ksize=(16, 1), op=op),
        ])

    def apply_pool(self, v):
        return np.concatenate([self.subapply(v, op=np.min),
                               self.subapply(v, op=np.max),
                               self.subapply(v, op=np.mean),
                               self.subapply(v, op=np.std)]).tolist()

    def rf_process(self, X_train, X_test, y_train, y_test):
        return (np.array([self.apply_pool(d) for d in X_train]),
                np.array([self.apply_pool(d) for d in X_test]),
                y_train,
                y_test)

    def get_rf_model(self):
        rf = RandomForestClassifier(random_state=101)
        return rf

    def train(self, save_path=None):
        self.model.fit(self.X_train, self.y_train)

    def test(self, identifier=None):
        pred = self.model.predict(self.X_test)
        conf_matrix = confusion_matrix(self.y_test, pred)
        class_report = classification_report(self.y_test, pred)
        f1 = f1_score(self.y_test, pred, average="weighted")
        result = "confusion matrix\n" + repr(
            conf_matrix) + "\n" + "report\n" + class_report + "\nf1_score(weighted)\n" + repr(f1)
        print(result)
        return {"cfm": conf_matrix, "report": class_report, "f1_score(weighted)": f1}


# ## Run Random Forest Classifier


frame_stack = 10
df = merged_df[['Amplitude', 'activity']]
data = df[['Amplitude']].values
label = df['activity'].values
label = convert_to_neumeric(label)
data, label = StackFrames(data, label, frame_stack)
label = np.array(list(map(lambda e: 1 if e == 9 else 0, label)))
X_train, X_test, y_train, y_test = split_dataset(data, label)


one = np.where(y_train == 1)[0]
not_one = np.random.choice(np.where(y_train == 0)[0], len(one))
new_index = np.array(one.tolist() + not_one.tolist())
X_train = X_train[new_index]
y_train = y_train[new_index]

rfModel = rf_model(X_train, X_test, y_train, y_test)
rfModel.train()
test_result = rfModel.test()


cfm = test_result['cfm']
total = cfm / cfm.sum(axis=1).reshape(-1, 1)
total = np.round(total, 2)
labels = ['Normal', 'Dengerous']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.show()
