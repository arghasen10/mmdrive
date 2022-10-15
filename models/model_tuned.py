import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split

final_activity_labels = ['Drinking', 'Fetching forward', 'Anomaly in Steering', 'Nodding', 'Yawning', 'Picking drops',
                         'Using Phone', 'Turning back', 'Talking left', 'Normal driving']


def get_df():
    merged_df = pd.read_pickle('../dataset/dataset_pub.pkl')
    merged_df.loc[merged_df["activity"] == "Talking back", "activity"] = "Turning back"
    merged_df.loc[merged_df["activity"] == "Talking Phone", "activity"] = "Using Phone"
    merged_df.loc[merged_df["activity"] == "Harsh driving", "activity"] = "Anomaly in Steering"
    return merged_df


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def convert_to_neumeric(label):
    lbl_map = \
        {'Drinking': 0,
         'Fetching forward': 1,
         'Anomaly in Steering': 2,
         'Nodding': 3,
         'Yawning': 4,
         'Picking drops': 5,
         'Using Phone': 6,
         'Turning back': 7,
         'Talking left': 8,
         'Normal driving': 9,
         }
    return np.array(list(map(lambda e: lbl_map[e], label)))


def split_dataset(data, label):
    np.random.seed(101)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test


def scale(doppz, Max=5336, Min=1375):
    doppz_scaled = (doppz - Min) / (Max - Min)
    return doppz_scaled


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
        return \
            np.concatenate([self.PoolOp(np.concatenate([np.expand_dims(e, 2) for e in v.transpose(1, 0)[2]], axis=2),
                                        ksize=(16, 16), op=op),
                            self.PoolOp(
                                np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[0]],
                                               axis=2), ksize=(16, 1), op=op),
                            self.PoolOp(
                                np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[1]],
                                               axis=2), ksize=(16, 1), op=op)])

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


def get_xtrain_ytrain(merged_df, frame_stack):
    frame_stack = 10
    df = merged_df[['rp_y', 'noiserp_y', 'doppz', 'activity']]
    data = df[['rp_y', 'noiserp_y', 'doppz']].values
    label = df['activity'].values
    label = convert_to_neumeric(label)
    mask = label != 9
    data, label = StackFrames(data[mask], label[mask], frame_stack)
    return split_dataset(data, label)


def plot_confusion_mat(test_result):
    cfm = test_result['cfm']
    total = cfm / cfm.sum(axis=1).reshape(-1, 1)
    total = np.round(total, 2)
    labels = ['Drinking', 'Fetching forward', 'Harsh driving', 'Nodding', 'Picking drops', 'Talking Phone',
              'Talking back', 'Talking left', 'Yawning']
    df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
    sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
    plt.show()


def _test_rf():
    df = get_df()
    X_train, X_test, y_train, y_test = get_xtrain_ytrain(df, frame_stack=10)
    rfModel = rf_model(X_train, X_test, y_train, y_test)
    rfModel.train()
    test_result = rfModel.test()
    print(test_result)


def get_cnn2d():
    model2d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu', input_shape=(16, 64, 10)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='cnn2d')
    return model2d


def get_cnn1d():
    model1d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (8, 2), (2, 1), padding="valid", activation='relu', input_shape=(64, 2, 10)),
        tf.keras.layers.Conv2D(64, (8, 1), (2, 1), padding="valid", activation='relu'),
        tf.keras.layers.Conv2D(96, (4, 1), (2, 1), padding="valid", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='cnn1d')
    return model1d


def get_dda_classifier():
    ann = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer='l2', input_shape=(224,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=9, activation='softmax')
    ], name='dda_classifier')
    return ann


def featureExtractor(X_1, X_2, X_3, cnn2d, cnn1d):
    X_23 = tf.keras.layers.Concatenate(axis=2)([X_2, X_3])
    emb1 = cnn2d(X_1)
    emb2 = cnn1d(X_23)
    return tf.keras.layers.Concatenate(axis=1)([emb1, emb2])


def get_feature_shape():
    X_1 = tf.keras.layers.Input(shape=(16, 64, 10))
    X_2 = tf.keras.layers.Input(shape=(64, 1, 10))
    X_3 = tf.keras.layers.Input(shape=(64, 1, 10))
    return X_1, X_2, X_3


def connect_feature_embeddings():
    cnn2d = get_cnn2d()
    cnn1d = get_cnn1d()
    dda = get_dda_classifier()
    x1, x2, x3 = get_feature_shape()
    # Connect
    emb = featureExtractor(x1, x2, x3, cnn2d, cnn1d)
    out_da = dda(emb)
    return x1, x2, x3, out_da


def get_fused_cnn_model():
    X_1, X_2, X_3, out_da = connect_feature_embeddings()
    model = tf.keras.Model(inputs=[X_1, X_2, X_3], outputs=[out_da], name='Fused_da_model')
    print(model.summary())
    return model


def preprocess_input_cnn(X_train):
    dop_train = np.array(
        [np.concatenate([np.expand_dims(e, 2) for e in v.transpose(1, 0)[2]], axis=2) for v in X_train])
    rp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[0]], axis=2) for v in X_train])
    noiserp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[1]], axis=2) for v in X_train])

    dop_train_s = (dop_train - dop_train.min()) / (dop_train.max() - dop_train.min())
    rp_train_s = (rp_train - rp_train.min()) / (rp_train.max() - rp_train.min())
    noiserp_train_s = (noiserp_train - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())
    return dop_train_s, rp_train_s, noiserp_train_s


def train_cnn(model, dop_train_s, rp_train_s, noiserp_train_s, y_train, epochs=3000):
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics="accuracy")

    history = \
        model.fit(
            [dop_train_s, rp_train_s, noiserp_train_s],
            y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
        )
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.show()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.show()
    return model


def test_cnn(model, dop_test_s, rp_test_s, noiserp_test_s, y_test):
    pred = model.predict([dop_test_s, rp_test_s, noiserp_test_s])
    class_report = classification_report(y_test, np.argmax(pred, axis=1))

    conf_matrix = confusion_matrix(y_test, np.argmax(pred, axis=1))
    return {"cfm": conf_matrix, "report": class_report}


def get_dvn_classifier():
    ann = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer='l2', input_shape=(224,)),
        tf.keras.layers.Dense(units=2, activation='softmax')
    ], name='dvn_classifier')
    return ann


def featureExtractor2(X_1, X_2, X_3, cnn2d, cnn1d):
    X_23 = tf.keras.layers.Concatenate(axis=2)([X_2, X_3])
    emb1 = cnn2d(X_1, training=False)
    emb2 = cnn1d(X_23, training=False)
    return tf.keras.layers.Concatenate(axis=1)([emb1, emb2])


def build_dvn_classifier_model():
    cnn2d = get_cnn2d()
    cnn1d = get_cnn1d()
    cnn2d.trainable = False
    cnn1d.trainable = False
    dvn = get_dvn_classifier()
    X_1, X_2, X_3 = get_feature_shape()
    emb = featureExtractor2(X_1, X_2, X_3, cnn2d, cnn1d)
    emb_dp = tf.keras.layers.Dropout(0.1)(emb)
    out_dn = dvn(emb_dp)

    model_dn = tf.keras.Model(inputs=[X_1, X_2, X_3], outputs=[out_dn], name='DN_model')
    model_dn.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4)
                     , metrics="accuracy")
    print(model_dn.summary())
    return model_dn


def removeOutlier(data, label):
    data1 = data[label == 1]
    label1 = label[label == 1]
    valid_idx = np.where(np.array(data1[:, :, 1].tolist()).reshape(-1, 640).mean(axis=1) <= 34.39010473)[0]
    data1 = data1[valid_idx]
    label1 = label1[valid_idx]

    data0 = data[label == 0]
    label0 = label[label == 0]

    r_ind = np.random.randint(0, data1.shape[0], data0.shape[0])
    data1 = data1[r_ind]
    label1 = label1[r_ind]

    Data = np.concatenate([data1, data0])
    Label = np.concatenate([label1, label0])
    return Data, Label


def prepare_inputs_dvn(df):
    df = df[['rp_y', 'noiserp_y', 'doppz', 'activity']]
    data = np.array(df[['rp_y', 'noiserp_y', 'doppz']].values.tolist())
    label = np.array(df['activity'].values.tolist())
    label = convert_to_neumeric(label)
    data, label = StackFrames(data, label, 10)
    newlabel = np.array([1 if e == 9 else 0 for e in label])
    Data, Label = removeOutlier(data, newlabel)
    X_train, X_test, y_train, y_test = split_dataset(Data, Label)

    not_one = np.where(y_train != 1)[0]
    one = np.random.choice(np.where(y_train == 1)[0], len(not_one))
    new_index = np.array(one.tolist() + not_one.tolist())
    X_train = X_train[new_index]
    y_train = y_train[new_index]
    dop_train_s, rp_train_s, noiserp_train_s = preprocess_input_cnn(X_train)
    dop_test_s, rp_test_s, noiserp_test_s = preprocess_input_cnn(X_test)
    return dop_train_s, rp_train_s, noiserp_train_s, dop_test_s, rp_test_s, noiserp_test_s, y_train, y_test


def fit_model_dvn(model_dn, dop_train_s, rp_train_s, noiserp_train_s, y_train, epochs=3000):
    history2 = \
        model_dn.fit(
            [dop_train_s, rp_train_s, noiserp_train_s],
            y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=256
        )

    plt.plot(history2.history['val_accuracy'], label='val_accuracy')
    plt.plot(history2.history['accuracy'], label='accuracy')
    plt.legend()
    plt.show()
    plt.plot(history2.history['loss'], label='loss')
    plt.plot(history2.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    return model_dn


def test_dvn(model_dn, dop_test_s, rp_test_s, noiserp_test_s, y_test):
    pred = model_dn.predict([dop_test_s, rp_test_s, noiserp_test_s])
    print(classification_report(y_test, np.argmax(pred, axis=1)))

    conf_matrix = confusion_matrix(y_test, np.argmax(pred, axis=1))

    cfm = conf_matrix
    total = cfm / cfm.sum(axis=1).reshape(-1, 1)
    total = np.round(total, 2)
    labels = ['Dangerous driving', 'Normal driving']
    df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
    sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
    plt.show()
