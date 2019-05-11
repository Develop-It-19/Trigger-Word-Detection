#Data Synthesis is an Effective way to Create Large Training Sets for Speech Problems.
  #Specifically for Trigger word detection.
#Using a Spectrogram and optionally a 1D conv layer is a common pre-processing step.
  #Prior to passing audio data to an RNN, GRU or LSTM.
#An End-to-end deep learning approach can be used to built a very effective trigger word detection system.

#Import Dependencies
import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
import matplotlib.pyplot as plt
%matplotlib inline

#We have raw_data directory containing:
  #Positive Examples of people saying "Activate".
  #Negative Examples of people saying Random words other than Activate.
  #10 Second clips of Background Noise.
IPython.display.Audio("./raw_data/activates/1.wav")
IPython.display.Audio("./raw_data/negatives/2.wav")
IPython.display.Audio("./raw_data/negatives/9.wav")

IPython.display.Audio("audio_examples/example_train.wav")
x = graph_spectrogram("audio_examples/example_train.wav")

_, data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)

Tx = 5511
n_freq = 101
Ty = 1375

#Load audio segments using pydub package
activates, negatives, background = load_raw_audio()
print("backgrounds[0] len: " + str(len(backgrounds[0])))
print("activate[0] len: " + str(len(activates[0])))
print("activate[1] len: " + str(len(activates[1])))

#Get a Random Time Segment from the 10 second audio.
def get_random_time_segment(segment_ms):
  segment_start = np.random.randint(low = 0, high = 10000 - segment_ms - 25)  #Lets take 25 to have atleast 3 consecutive time steps after the end of activate timesteps if it is applied at the very end we need some time steps to label. 
  segment_end = segment_start + segment_ms - 1
  
  return (segment_start, segment_end)

#Check if the new random time segment overlaps with the previous segments.
def is_overlapping(segment_time, previous_segments):
  segment_start, segment_end = segment_time
  overlap = False
  
  for previous_start, previous_end in previous_segments:
    if segment_start <= previous_end and segment_end >= previous_start:
      overlap = True
  
  return overlap

overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
print("Overlap 1 = ", overlap1)
print("Overlap 2 = ", overlap2)

#Insert an Audio clip over the Background such that it doesn't overlap with previously inserted Audio clips.
def insert_audio_clip(background, audio_clip, previous_segments):
  segment_ms = len(audio_clip)
  segment_time = get_random_time_segment(segment_ms)
  
  while is_overlapping(segment_time, previous_segments):
    segment_time = get_random_time_segment(segment_ms)
  
  previous_segments.append(segment_time)
  
  new_background = background.overlay(audio_clip, position = segment_time[0])
  
  return new_background, segment_time

np.random.seed(7)
new_background_with_audio_clip, segment_time = insert_audio_clip(background[0], activates[0], [(1328, 2795)])
new_background_with_audio_clip.export("insert_test.wav", format = "wav")
print("Segment Time: ", segment_time)
IPython.display.Audio("insert_test.wav")
IPython.display.Audio("audio_examples/insert_reference.wav")

#Update the labels with ones after the end of the word "Activate".
def insert_ones(y, segment_end_ms):
  segment_end_y = int(segment_end_ms * Ty / 10000.0)
  
  for i in range(segment_end_y + 1, segment_end_y + 51):
    if i < Ty:
      y[0, i] = 1
  
  return y

arr1 = insert_ones(np.zeros((1, Ty)), 9700)
plt.plot(insert_ones(arr1, 4251)[0,:])
print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])

#Generate a Training Example
def create_training_example(background, activates, negatives):
  np.random.seed(3)
  
  background = background - 20  #Make the Background Quieter
  
  y = np.zeros((1, Ty))
  previous_segments = []
  
  number_of_activates = np.random.randint(0, 5)
  random_indices = np.random.randint(len(activates), size = number_of_activates)
  random_activates = [activates[i] for i in random_indices]
  
  for random_activate in random_activates:
    background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
    segment_start, segment_end = segment_time
    y = insert_ones(y, segment_end_ms = segment_end)
  
  number_of_negatives = np.random.randint(0, 3)
  random_indices = np.random.randint(len(negatives), size = number_of_negatives)
  random_negatives = [negative[i] for i in random_indices]
  
  for random_negative in random_negatives:
    background, _ = insert_audio_clip(background, random_negative, previous_segments)
  
  background = match_target_amplitude(background, -20.0)  #Standardize the Volume of the Audio clip.
  
  file_handle = background.export("train" + ".wav", format="wav")
  print("File (train.wav) was saved in your directory.")
  
  x = graph_spectrogram("train.wav")
  
  return x, y

x, y = create_training_example(background[0], activates, negatives)

IPython.display.Audio("train.wav")
IPython.display.Audio("audio_examples/train_reference.wav")

plt.plot(y[0])

#Load the Synthesized Training Dataset
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

#Load the Real Dev Dataset.
  #Real because the Dev set should come from the same distribution as the Test set.
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

#Model
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

def model(input_shape):
  X_input = Input(shape = input_shape)
  
  X = Conv1D(196, kernel_size = 15, strides = 4)(X_input)
  X = BatchNormalization()(X)
  X = Activation("relu")(X)
  X = Dropout(0.8)(X)
  
  X = GRU(units = 128, return_sequences = True)(X)
  X = Dropout(0.8)(X)
  X = BatchNormalization()(X)
  
  X = GRU(units = 128, return_sequences = True)(X)
  X = Dropout(0.8)(X)
  X = BatchNormalization()(X)
  X = Dropout(0.8)(X)
  
  X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)
  
  model = Model(inputs = X_input, outputs = X)
  
  return model

model = model(input_shape = (Tx, n_freq))

model.summary()

model = load_model("./models/tr_model.h5")

opt = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

model.fit(X, Y, batch_size = 5, epochs = 1)

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)

def detect_triggerword(filename):
  plt.subplot(2, 1, 1)
  
  x = graph_spectrogram(filename)
  x = x.swapaxes(0, 1)
  x = np.expand_dims(x, axis = 0)
  predictions = model.predict(x)
  
  plt.subplot(2, 1, 2)
  plt.plot(predictions[0, :, 0])
  plt.ylabel('probability')
  plt.show()
  
  return predictions

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
  audio_clip = AudioSegment.from_wav(filename)
  chime = AudioSegment.from_wav(chime_file)
  Ty = predictions.shape[1]
  
  consecutive_timesteps = 0
  
  for i in range(Ty):
    consecutive_timesteps += 1
    
    if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
      audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
      consecutive_timesteps = 0
  
  audio_clip.export("chime_output.wav", format = "wav")

IPython.display.Audio("./raw_data/dev/1.wav")
IPython.display.Audio("./raw_data/dev/2.wav")

filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activation(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")

filename = "./raw_data/dev/2.wav"
prediction = detect_triggerword(filename)
chime_on_activation(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")

def preprocess_audio(filename):
  padding = AudioSegment.silent(duration = 10000)
  segment = AudioSegment.from_wav(filename)[:10000]
  segment = padding.overlay(segment)
  segment = segment.set_frame_rate(44100)
  segment.export(filename, format = "wav")

your_filename = "audio_examples/my_audio.wav"

preprocess_audio(your_filename)
IPython.display.Audio(your_filename)

chime_threshold = 0.5
prediction = detect_triggerword(your_filename)
chime_on_activate(your_filename, prediction, chime_threshold)
IPython.display.Audio("./chime_output.wav")

