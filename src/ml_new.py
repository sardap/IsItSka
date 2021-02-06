import os
import pathlib
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from env import SKA_PLAYLIST, NOT_SKA_PLAYLIST, CLF_FOLDER_PATH
from spotify_helper import get_playlist_by_name, get_playlist_tracks, get_analysis_for_tracks, get_features_for_tracks
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.convolutional import Convolution2D
import glob
import pickle


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def fill_blanks(x, n):
	if len(x) > n:
		return x[:n]
	
	for i in range(n - len(x), n):
		x.append(-1)

	return x

def get_durations_from_beats(x):
	return [i["duration"] for i in x]

def calc_avg(x, key):
	s = 0
	for i in x:
		s += i[key]

	return s / len(x)

def merge_dict(x, y):

	for i in range(len(x)):
		x[i]["num_samples"] = y[i]["track"]["num_samples"]
		# x[i]["rhythmstring"] = fill_blanks(y[i]["track"]["rhythmstring"], 1000)
		# x[i]["beat_duration_avg"] = calc_avg(y[i]['beats'], 'duration')#fill_blanks(get_durations_from_beats(y[i]["beats"]), 50)
			
	return x

def get_data():
	random.seed(200)

	play_id, length = get_playlist_by_name(SKA_PLAYLIST)
	length = 10
	track_ids = get_playlist_tracks(play_id=play_id, n=length)
	track_ids = track_ids
	ska_complete = merge_dict(
		get_features_for_tracks(track_ids),
		get_analysis_for_tracks(track_ids))
	
	play_id, length = get_playlist_by_name(NOT_SKA_PLAYLIST)
	length = 10
	track_ids = get_playlist_tracks(play_id=play_id, n=length)
	random.shuffle(track_ids)
	track_ids = track_ids[:len(ska_complete)]
	not_ska_complete = merge_dict(
		get_features_for_tracks(track_ids),
		get_analysis_for_tracks(track_ids))

	complete = []
	for track in ska_complete:
		track["ska"] = True
		complete.append(track)

	for track in not_ska_complete:
		track["ska"] = False
		complete.append(track)

	return complete

def create_model():
	# new_model = tf.keras.models.load_model(os.path.join(CLF_FOLDER_PATH, "clf.tf"))

	# new_model.summary()

	complete = get_data()
	
	dataframe = pd.DataFrame.from_dict(complete)

	dataframe['target'] = np.where(dataframe['ska'], 1, 0)

	# Drop un-used columns.
	dataframe = dataframe.drop(columns=[
		'analysis_url', 'track_href', 'uri', 
		'id', 'type', 'energy', 'danceability', 'speechiness', 'acousticness', 'liveness',
	])

	feature_columns = []

	for header in ['loudness', 'tempo']: #, 'key', 'mode', 'tempo', 'duration_ms', 'time_signature', 'beat_duration_avg']:
		feature_columns.append(feature_column.numeric_column(header))

	# bucket = feature_column.bucketized_column(
	# 	feature_column.numeric_column('tempo'), boundaries=[0, 40, 60, 66, 76, 120, 168, 200])
	# feature_columns.append(bucket)


	for header in ['instrumentalness', 'valence']:
		bucket = feature_column.bucketized_column(
			feature_column.numeric_column(header), boundaries=[0, 0.25, 0.5, 0.75, 1])
		feature_columns.append(bucket)

	train, test = train_test_split(dataframe, test_size=0.2)
	train, val = train_test_split(train, test_size=0.2)
	print(len(train), 'train examples')
	print(len(val), 'validation examples')
	print(len(test), 'test examples')

	batch_size = 32
	train_ds = df_to_dataset(train, batch_size=batch_size)
	val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
	test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

	feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

	model = tf.keras.Sequential()

	model.add(feature_layer)
	model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(10))
	model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
	# model = tf.keras.Sequential([
	# 	feature_layer,
	# 	tf.keras.layers.Flatten(input_shape=(28, 28)),
	# 	tf.keras.layers.Dense(128, activation='relu'),
	# 	tf.keras.layers.Dense(10)
	# ])

	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	model.fit(train_ds,
		validation_data=val_ds,
		epochs=10)


	model.build(input_shape=(1,28,28))

	loss, accuracy = model.evaluate(test_ds)
	print("Accuracy", accuracy)

	# pickle.dump(model, open(os.path.join(CLF_FOLDER_PATH, "clf.bin"), "wb"))
	model.save(os.path.join(CLF_FOLDER_PATH, "clf.h5"))

def load_model():
	new_model = tf.keras.models.load_model(os.path.join(CLF_FOLDER_PATH, "clf.h5"))
	#pickle.load(open(os.path.join(CLF_FOLDER_PATH, "clf.bin"), "rb"))
	# Check its architecture
	new_model.summary()


def start_ml():
	create_model()
	load_model()