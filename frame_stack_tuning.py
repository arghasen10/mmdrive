import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split

merged_df = pd.read_pickle('../merged_dataset.pkl')
merged_df.loc[merged_df["activity"] == "Talking back", "activity"] = "Turning back"
merged_df.loc[merged_df["activity"] == "Talking Phone", "activity"] = "Using Phone"
merged_df.loc[merged_df["activity"] == "Harsh driving", "activity"] = "Anomaly in Steering"


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
         'Normal driving': 9}
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


def get_cnn2d():
    model2d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu',
                               input_shape=(16, 64, frame_stack)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='cnn2d')
    return model2d


def get_cnn1d():
    model1d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (8, 2), (2, 1), padding="valid", activation='relu',
                               input_shape=(64, 2, frame_stack)),
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
    emb1 = cnn2d(X_1);
    emb2 = cnn1d(X_23)
    return tf.keras.layers.Concatenate(axis=1)([emb1, emb2])


def provide_model(frame_stack):
    X_1 = tf.keras.layers.Input(shape=(16, 64, frame_stack))
    X_2 = tf.keras.layers.Input(shape=(64, 1, frame_stack))
    X_3 = tf.keras.layers.Input(shape=(64, 1, frame_stack))

    cnn2d = get_cnn2d()
    cnn1d = get_cnn1d()
    dda = get_dda_classifier()

    # Connect
    emb = featureExtractor(X_1, X_2, X_3, cnn2d, cnn1d)
    # isnormal=tf.cast(dvn(tf.stop_gradient(emb))>0.5,tf.float32)
    out_da = dda(emb)
    # out=tf.concat([tf.multiply(da,1-isnormal),isnormal],axis=1)

    model = tf.keras.Model(inputs=[X_1, X_2, X_3], outputs=[out_da], name='Fused_da_model')
    return model


def frame_exp(frame_stack):
    print('Experiment with frame no ', frame_stack)
    df = merged_df[['rp_y', 'noiserp_y', 'doppz', 'activity']]
    data = np.array(df[['rp_y', 'noiserp_y', 'doppz']].values.tolist())
    label = np.array(df['activity'].values.tolist())
    label = convert_to_neumeric(label)

    data, label = StackFrames(data, label, frame_stack)
    mask = label != 9
    X_train, X_test, y_train, y_test = split_dataset(data[mask], label[mask])
    dop_train = np.array(
        [np.concatenate([np.expand_dims(e, 2) for e in v.transpose(1, 0)[2]], axis=2) for v in X_train])
    rp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[0]], axis=2) for v in X_train])
    noiserp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[1]], axis=2) for v in X_train])

    dop_train_s = (dop_train - dop_train.min()) / (dop_train.max() - dop_train.min())
    rp_train_s = (rp_train - rp_train.min()) / (rp_train.max() - rp_train.min())
    noiserp_train_s = (noiserp_train - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())
    model = provide_model(frame_stack)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
                  # tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics="accuracy")
    history = \
        model.fit(
            [dop_train_s, rp_train_s, noiserp_train_s],
            y_train,
            epochs=1000,
            validation_split=0.2,
            batch_size=32,
        )
    dop_test = np.array([np.concatenate([np.expand_dims(e, 2) for e in v.transpose(1, 0)[2]], axis=2) for v in X_test])
    rp_test = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[0]], axis=2) for v in X_test])
    noiserp_test = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[1]], axis=2) for v in X_test])

    dop_test_s = (dop_test - dop_train.min()) / (dop_train.max() - dop_train.min())
    rp_test_s = (rp_test - rp_train.min()) / (rp_train.max() - rp_train.min())
    noiserp_test_s = (noiserp_test - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())
    pred = model.predict([dop_test_s, rp_test_s, noiserp_test_s])
    all_f1 = classification_report(y_test, np.argmax(pred, axis=1)).strip().split('\n')
    ddb_f1 = np.asarray([e.strip().split()[3] for e in all_f1[2:-4]], dtype=float).tolist()
    return {'frames': frame_stack,
            'accuracy': ddb_f1,
            'overall': float(all_f1[-1].split()[-2])}


for frame_stack in np.linspace(1, 80, 20, dtype=int):
    dict_res = frame_exp(frame_stack)
    with open('logger_frames.json', 'a') as file:
        file.write(json.dumps(dict_res))
