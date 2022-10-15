import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from helper import convert_to_neumeric, split_dataset, scale, get_df


def StackFrames(doppz, labels, frame_stack=4):
    max_index = doppz.shape[0] - frame_stack
    stacked_doppz = np.array([doppz[i:i + frame_stack] for i in range(max_index)]).transpose(0, 2, 3, 1)
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_doppz, new_labels


def get_aug_pipe(frame_stack=10):
    Input = tf.keras.layers.Input(shape=(16, 64, frame_stack))
    net = tf.keras.layers.Resizing(height=48, width=48, name='Resize_48x48')(Input)
    pipe = tf.keras.Model(inputs=[Input], outputs=[net], name='Aug_pipe')
    return pipe


def preprocess_vgg(df, frame_stack):
    doppz = np.array(df['doppz'].values.tolist())
    label = df['activity'].values
    doppz_scaled_stacked, new_labels = StackFrames(scale(doppz, doppz.max(), doppz.min()), label, frame_stack)
    pipe = get_aug_pipe(frame_stack=frame_stack)
    doppz_aug = pipe(doppz_scaled_stacked, training=False).numpy()
    return doppz_aug, new_labels


class vgg16_model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = self.vgg16_process(X_train, X_test, y_train, y_test)
        self.model = self.get_vgg16_model()

    def vgg16_process(self, X_train, X_test, y_train, y_test):
        return (preprocess_input(np.uint8(X_train[:, :, :, -3:] * 255)),
                preprocess_input(np.uint8(X_test[:, :, :, -3:] * 255)), y_train, y_test)

    def get_vgg16_model(self):
        tf.random.set_seed(101)
        vgg16_topless = VGG16(weights="vgg_weights.h5", include_top=False)
        vgg16_topless.trainable = False
        model = tf.keras.Sequential([
            vgg16_topless,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(9, "softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics="accuracy")
        return model

    def train(self, epochs=1000, save_path=None):
        self.model.fit(self.X_train, self.y_train,
                       epochs=epochs,
                       validation_split=0.2,
                       batch_size=32)  # ,

    #                     callbacks=[best_save])

    def test(self, identifier=None):
        pred = np.argmax(self.model.predict(self.X_test), axis=1)
        conf_matrix = confusion_matrix(self.y_test, pred)
        class_report = classification_report(self.y_test, pred)
        f1 = f1_score(self.y_test, pred, average="weighted")
        result = "confusion matrix\n" + repr(
            conf_matrix) + "\n" + "report\n" + class_report + "\nf1_score(weighted)\n" + repr(f1)
        print(result)


if __name__ == "__main__":
    frame_stack = 10
    df = get_df()
    df = df[['doppz', 'activity']]
    data, label = preprocess_vgg(df, frame_stack)
    label = convert_to_neumeric(label)
    mask = label != 9
    X_train, X_test, y_train, y_test = split_dataset(data[mask], label[mask])
    vgg = vgg16_model(X_train, X_test, y_train, y_test)
    vgg.train(epochs=1)
    vgg.test()
