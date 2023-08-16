#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import glob
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier


# ### Helper Functions

# In[2]:


def convert_to_neumeric(label):
    lbl_map = \
    {'Drinking':0, 
     'Fetching forward':1, 
     'Harsh driving':2, 
     'Nodding':3,
     'Yawning':4,
     'Picking drops':5, 
     'Talking Phone':6, 
     'Talking back':7,
     'Talking left':8,
     'Normal driving':9,
     }
    return np.array(list(map(lambda e: lbl_map[e], label)))

def StackFrames(data, labels, frame_stack=5):
    max_index = data.shape[0] - frame_stack
    stacked_data = np.array([data[i:i + frame_stack] for i in range(max_index)])
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_data, new_labels


# In[3]:


np.random.seed(101)


# ### Importing Dataset

# In[4]:


# df = pd.read_pickle('new_df2.pkl')
# df['datetime'] = df['datetime'].apply(pd.to_datetime)
# df = df.sort_values(by='datetime', ascending=True).reset_index(drop=True)
# dang_activities = ['Drinking', 'Fetching forward', 'Harsh driving', 'Nodding', 'Picking drops', 'Talking Phone', 'Talking back', 'Talking left', 'Yawning']


# ### Finding chunks

# In[5]:


# chunks=[]
# prev_act='Normal driving'
# prev_i=0
# i=0
# curr_dang_activity = None
# while i<df.shape[0]:
#     i+=1
#     while True and i < df.shape[0]:
#         prev_act = df.iloc[i-1].activity
#         if df.iloc[i].activity != 'Normal driving':
#             if curr_dang_activity == None:
#                 curr_dang_activity = df.iloc[i].activity
#             if curr_dang_activity != df.iloc[i].activity:
#                 curr_dang_activity = df.iloc[i].activity
#                 break
#         if df.iloc[i].datetime-df.iloc[i-1].datetime <= timedelta(seconds=1) and \
#         (df.iloc[i].activity == prev_act or df.iloc[i].activity == 'Normal driving' or \
#          df.iloc[i-1].activity == 'Normal driving'):
#             i+=1
#         else:
#             break
#     chunks.append(df.iloc[prev_i:i])
#     prev_i=i

# prev_dang_activity = None
# new_chunks = []
# i = 0
# prev_i = 0
# for c in chunks:
#     i+=1
#     for act in c.groupby('activity').size().keys():
#         if act in dang_activities:
#             if prev_dang_activity == None:
#                 prev_dang_activity = act
#             if prev_dang_activity != act or (pd.to_datetime(chunks[i].datetime.values[0]) - pd.to_datetime(chunks[i-1].datetime.values[-1])).days > 1:
#                 prev_dang_activity = act
#                 new_chunks.append(pd.concat(chunks[prev_i:i-1], ignore_index=True))
#                 prev_i = i-1
                
# print(len(new_chunks))


# ### train test datasets

# In[6]:


# train_df = []
# test_df = []
# train_test_split = 0.8
# for c in new_chunks:
#     train_df.append(c.iloc[:int(len(c)*train_test_split)])
#     test_df.append(c.iloc[int(len(c)*train_test_split):])


# In[7]:


# train_df = pd.concat(train_df, ignore_index=True)
# test_df = pd.concat(test_df, ignore_index=True)


# In[8]:


train_df = pd.read_pickle('train_df.pkl')
test_df = pd.read_pickle('test_df.pkl')


# ### Helper Functions for Random Forest Classifer

# In[9]:


class rf_model:
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test=self.rf_process(X_train, X_test, y_train, y_test)
        self.model=self.get_rf_model()

    def PoolOp(self,img, ksize=(16, 16), op=np.mean):
        i_h, i_w, i_c = img.shape
        k_h, k_w = ksize
        row = []
        for c in range(i_c):
            for i in range(i_h // k_h):
                for j in range(i_w // k_w):
                    row.append(op(img[k_h * i:k_h * i + k_h, k_w * j:k_w * j + k_w, c]))
        return np.array(row)
    
    def subapply(self,v,op):
        return \
        np.concatenate([self.PoolOp(np.concatenate([np.expand_dims(e,2) for e in v.transpose(1,0)[2]],axis=2),ksize=(16, 16),op=op),
        self.PoolOp(np.concatenate([np.expand_dims(e.reshape(-1,1),2) for e in v.transpose(1,0)[0]],axis=2),ksize=(16, 1),op=op),
        self.PoolOp(np.concatenate([np.expand_dims(e.reshape(-1,1),2) for e in v.transpose(1,0)[1]],axis=2),ksize=(16, 1),op=op)])

    def apply_pool(self,v):
        return np.concatenate([self.subapply(v, op=np.min),
                               self.subapply(v, op=np.max),
                               self.subapply(v, op=np.mean),
                               self.subapply(v, op=np.std)]).tolist()

    def rf_process(self,X_train, X_test, y_train, y_test):
        return (np.array([self.apply_pool(d) for d in X_train]),
            np.array([self.apply_pool(d) for d in X_test]),
            y_train,
            y_test)

    def get_rf_model(self):
        rf = RandomForestClassifier(random_state=101)
        return rf

    def train(self,save_path=None):
        self.model.fit(self.X_train,self.y_train)
    
    def test(self,identifier=None):
        pred=self.model.predict(self.X_test)
        conf_matrix=confusion_matrix(self.y_test,pred)
        class_report=classification_report(self.y_test,pred)
#         fpr, fnr, thresholds = det_curve(self.y_test, pred)
        f1=f1_score(self.y_test,pred,average="weighted")
        result="confusion matrix\n"+repr(conf_matrix)+"\n"+"report\n"+class_report+"\nf1_score(weighted)\n"+repr(f1)
        print(result)
        return {"cfm":conf_matrix, "report": class_report, "f1_score(weighted)": f1}


# ## Helper Functions for CNN Classifier

# In[10]:


def get_cnn2d():
    model2d=tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,(2,5),(1,2),padding="same",activation='relu',input_shape=(16,64,5)),
                tf.keras.layers.Conv2D(64,(2,3),(1,2),padding="same",activation='relu'),
                tf.keras.layers.Conv2D(96,(3,3),(2,2),padding="same",activation='relu'),
                tf.keras.layers.Conv2D(128,(3,3),(2,2),padding="same",activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
#                 tf.keras.layers.Dropout(rate=0.25)
            ],name='cnn2d')
    return model2d


def get_cnn1d():
    model1d=tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,(8,2),(2,1),padding="valid",activation='relu',input_shape=(64,2,5)),
                tf.keras.layers.Conv2D(64,(8,1),(2,1),padding="valid",activation='relu'),
                tf.keras.layers.Conv2D(96,(4,1),(2,1),padding="valid",activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
#                 tf.keras.layers.Dropout(rate=0.3)
            ],name='cnn1d')
    return model1d

def get_dda_classifier():
    ann=tf.keras.Sequential([
                tf.keras.layers.Dense(units=64,activation='relu', kernel_regularizer='l2', input_shape=(224,)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(units=32,activation='relu', kernel_regularizer='l2'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(units=9,activation='softmax')
            ],name='dda_classifier')
    return ann

def featureExtractor(X_1,X_2,X_3,cnn2d,cnn1d):
    X_23=tf.keras.layers.Concatenate(axis=2)([X_2,X_3])
    emb1=cnn2d(X_1);emb2=cnn1d(X_23)
    return tf.keras.layers.Concatenate(axis=1)([emb1,emb2])


# In[11]:


X_train = np.array([row[:3] for row in train_df[['rp_y', 'noiserp_y', 'doppz']].values])
y_train = np.array(train_df['activity'].values.tolist())
y_train = convert_to_neumeric(y_train)
X_train, y_train = StackFrames(X_train, y_train, frame_stack=5)

X_test = np.array([row[:3] for row in test_df[['rp_y', 'noiserp_y', 'doppz']].values])
y_test = np.array(test_df['activity'].values.tolist())
y_test = convert_to_neumeric(y_test)
X_test, y_test = StackFrames(X_test, y_test, frame_stack=5)

mask1=y_train!=9
mask2=y_test!=9
X_train, X_test, y_train, y_test = X_train[mask1], X_test[mask2], y_train[mask1], y_test[mask2]


# In[12]:


rfModel = rf_model(X_train, X_test, y_train, y_test)
rfModel.train()
test_result = rfModel.test()
cfm = test_result['cfm']
total = cfm / cfm.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['Drinking', 'Fetching forward', 'Harsh driving', 'Nodding', 'Yawning', 'Picking drops', 'Talking Phone', 'Talking back', 'Talking left']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")


# ### Data PreProcessing

# In[13]:


dop_train = np.array([np.concatenate([np.expand_dims(e,2) for e in v.transpose(1,0)[2]],axis=2) for v in X_train])
rp_train = np.array([np.concatenate([np.expand_dims(e.reshape(-1,1),2) for e in v.transpose(1,0)[0]],axis=2) for v in X_train])
noiserp_train = np.array([np.concatenate([np.expand_dims(e.reshape(-1,1),2) for e in v.transpose(1,0)[1]],axis=2) for v in X_train])

dop_train_s = (dop_train - dop_train.min()) / (dop_train.max() - dop_train.min())
rp_train_s = (rp_train - rp_train.min()) / (rp_train.max() - rp_train.min())
noiserp_train_s = (noiserp_train - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())


# In[14]:


X_1=tf.keras.layers.Input(shape=(16,64,5))
X_2=tf.keras.layers.Input(shape=(64,1,5))
X_3=tf.keras.layers.Input(shape=(64,1,5))

###Sub-modules
cnn2d=get_cnn2d()
cnn1d=get_cnn1d()
dda=get_dda_classifier()

#Connect
emb=featureExtractor(X_1,X_2,X_3,cnn2d,cnn1d)
# isnormal=tf.cast(dvn(tf.stop_gradient(emb))>0.5,tf.float32)
out_da=dda(emb)
# out=tf.concat([tf.multiply(da,1-isnormal),isnormal],axis=1)

model=tf.keras.Model(inputs=[X_1,X_2,X_3],outputs=out_da,name='Fused_da_model')


# In[15]:


model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',#tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics="accuracy")


# In[16]:


history=model.fit(
    [dop_train_s, rp_train_s, noiserp_train_s],
    y_train.reshape(-1,1),
    epochs=10000,
    validation_split=0.2,
    batch_size=32
)


# In[ ]:


plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[ ]:


dop_test = np.array([np.concatenate([np.expand_dims(e,2) for e in v.transpose(1,0)[2]],axis=2) for v in X_test])
rp_test = np.array([np.concatenate([np.expand_dims(e.reshape(-1,1),2) for e in v.transpose(1,0)[0]],axis=2) for v in X_test])
noiserp_test = np.array([np.concatenate([np.expand_dims(e.reshape(-1,1),2) for e in v.transpose(1,0)[1]],axis=2) for v in X_test])

dop_test_s = (dop_test - dop_train.min()) / (dop_train.max() - dop_train.min())
rp_test_s = (rp_test - rp_train.min()) / (rp_train.max() - rp_train.min())
noiserp_test_s = (noiserp_test - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())


# In[ ]:


pred=model.predict([dop_test_s, rp_test_s, noiserp_test_s])
print(classification_report(y_test,np.argmax(pred,axis=1)))

conf_matrix = confusion_matrix(y_test, np.argmax(pred, axis=1))

cfm = conf_matrix
total = cfm / cfm.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['Drinking', 'Fetching\nforward', 'Anomaly\nin Steering', 'Nodding', 'Yawning', 'Picking\ndrops',
          'Using\nPhone', 'Turning\nback', 'Talking\nleft']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")


# In[ ]:




