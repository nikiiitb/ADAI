# %%
from datasets import load_dataset
from huggingface_hub import from_pretrained_keras
import cv2

import os
os.environ['CURL_CA_BUNDLE'] = ''

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

# %%
# tf.config.gpu.set_per_process_memory_growth(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu , True)

# %% [markdown]
# ### Loading the Video Swin Model

# %%
swin_model = from_pretrained_keras("Tonic/video-swin-transformer")

# %%
swin_model.summary()

# %% [markdown]
# ### Frame Analysis

# %%
# vid = cv2.VideoCapture("./SSBD/videos/v_ArmFlapping_01.avi")
# ret, frame = vid.read()

# plt.imshow(frame)

frame_lengths = []
for i in range(49):
    cap = cv2.VideoCapture(f"./inputs/clipped/armflapping/{i}.avi")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("Number of frames : " , length)
    frame_lengths.append(length)
    
frame_lengths = pd.DataFrame(np.array(frame_lengths))
frame_lengths.describe()
# sorted(frame_lengths)

# %% [markdown]
# ### Testing the video swin transformer

# %%
idx = 3 #video number

frames = []
video = cv2.VideoCapture(f"./inputs/clipped/armflapping/{idx}.avi")
while True:
    read, frame= video.read()
    if not read:
        break
    frames.append(frame)
frames = np.array(frames)

frames.shape

# %%
CLIP_SIZE = 32
# BATCH_SIZE = 1
frame_size = frames.shape[0]
num_batches = (frame_size // 32)
clipped_frames_size = num_batches * 32
clipped_frames = frames[:clipped_frames_size]
clipped_frames = clipped_frames.reshape(num_batches , 32 , 224 , 224 , 3)
clipped_frames = np.rollaxis(clipped_frames, 4, 1)
# # clipped_frames = clipped_frames[np.newaxis , ...]
# clipped_frames = clipped_frames.astype(np.float32)
clipped_frames = clipped_frames.astype(np.uint8)
clipped_frames = np.average(clipped_frames , axis=0)
print("Video Averaged every 32 frames : ", clipped_frames.shape)

# %%
outp = swin_model.predict(clipped_frames[np.newaxis , ...])
# outp = swin_model(clipped_frames)
outp.shape , type(outp)

# %%
outp.max() , outp.min()

# %%
NUM_CLASSES = 3

def build_model():
    inp = tf.keras.layers.Input(shape = (3,32,224,224))
    x = swin_model(inp)
    x = tf.keras.layers.GlobalAveragePooling3D(data_format="channels_first")(x)
    # x = tf.keras.layers.Flatten(data_format="channels_first")(x)
    x = tf.keras.layers.Dense(units = 64 , activation='relu' , input_shape=(1024,))(x)
    x = tf.keras.layers.Dense(units = NUM_CLASSES , activation='softmax')(x)
    
    model = tf.keras.Model(inputs = [inp] , outputs = x)
    return model

modelx = build_model()
opt = tf.keras.optimizers.AdamW(3e-5 , 1e-2)
# opt = Adam(3e-5)
modelx.compile(optimizer= opt, loss="categorical_crossentropy", metrics=['accuracy'])
modelx.summary()

# %% [markdown]
# ### Creating Dataset

# %%
DATA = [
    # DIRECTORY , CLASS
    ("./inputs/clipped/armflapping" , 0),
    ("./inputs/clipped/headbanging" , 1),
    ("./inputs/clipped/spinning" , 2)
]

DATA_32 = [] #contains videos more than 32 frames

for data in DATA:
    vids = [f for f in os.listdir(data[0]) if os.path.isfile(os.path.join(data[0], f))]
    for vid in vids:
        cap_dir = f"{data[0]}/{vid}"
        cap = cv2.VideoCapture(cap_dir)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length >= 32:
            DATA_32.append((cap_dir , data[1])) #directory of video and class of video
            
df = pd.DataFrame(DATA_32 , columns=['video' , 'label'])
df.head()

# %%
class Dataset_Generator(tf.keras.utils.Sequence) :
    def __init__(self, videos, labels, batch_size) :
        self.videos = videos
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self) :
        return int(len(self.videos) / self.batch_size)

    def __getitem__(self, idx) :
        batch_x = self.videos[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        
        vids = []
        labels = []
        for vid , label in zip(batch_x , batch_y):
            video = cv2.VideoCapture(vid)
            frames = []
            while True:
                read, frame= video.read()
                if not read:
                    break
                frames.append(frame)
            frames = np.array(frames)
            frames = frames[:(len(frames) // 32) * 32]
            
            frame_size = frames.shape[0]
            num_batches = (frame_size // 32)
            clipped_frames_size = num_batches * 32
            clipped_frames = frames[:clipped_frames_size]
            clipped_frames = clipped_frames.reshape(num_batches , 32 , 224 , 224 , 3)
            clipped_frames = np.rollaxis(clipped_frames, 4, 1)
            clipped_frames = clipped_frames.astype(np.float32)
            clipped_frames = np.average(clipped_frames , axis=0)
            vids.append(clipped_frames)
            # print(clipped_frames.shape)
            
            l = [0 for i in range(NUM_CLASSES)]
            l[label] = 1
            labels.append(l)

        return np.array(vids).astype(np.uint8) , np.array(labels)

# %%
BATCH_SIZE = 4

validation_split = 0.3
# validation_split_len = int(validation_split * SAMPLES)

# df_sampled = df.sample(frac=1).reset_index(drop=True) #shuffle the dataset

# df_train = df_sampled[: -validation_split_len]
# df_test = df_sampled[-validation_split_len : - validation_split_len // 2]
# df_val = df_sampled[- validation_split_len // 2 : ]

X = df["video"]
Y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_split, random_state=42)

validation_test_split = len(X_test) // 2

training_gen = Dataset_Generator(
    X_train.to_list(), 
    y_train.to_list(),
    BATCH_SIZE
)

validation_gen = Dataset_Generator(
    X_test[:validation_test_split].to_list(), 
    y_test[:validation_test_split].to_list(),
    BATCH_SIZE
)

test_gen = Dataset_Generator(
    X_test[validation_test_split:].to_list(), 
    y_test[validation_test_split:].to_list(),
    BATCH_SIZE
)

len(X_train) , len(X_test)

# %%
x , y = training_gen.__getitem__(0)
x.shape

# %%
from sys import getsizeof

print("Size of one batch : " , getsizeof(x) // 1024 // 1024 , "MBs")

# %% [markdown]
# ### Training the Model

# %%
# modelx(clipped_frames)

modelx.fit(training_gen , epochs=5 , validation_data=validation_gen)

# %%



