import os
import multiprocessing
import graphviz
import pickle
import io
import itertools
import hashlib
import re
import random

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score
from sklearn import metrics, tree
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from datetime import datetime

from env import API_AUTH, REFRESH_TOKEN, SOURCE_PLAYLIST, SKA_PLAYLIST
from spotify_helper import get_analysis_for_tracks, get_access_token, get_playlist_tracks, get_genres_for_track, set_login, get_analysis_for_track, get_track, add_tracks_to_playlist, get_playlist_by_name


os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"

MIN_CONFIDENC1E = 0.7

FEATURES = [
	"beats"
]

class ClfInfo():
	clf = None
	n_beats = None
	n_sections = None
	n_segments = None
	n_tatums = None
	feature_set = None
	length_dict = None

def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def transform(complete_track, key, other_key, n, result):
	for i in range(0, n):
		if i < len(complete_track["analysis"][key]):
			val = float(complete_track["analysis"][key][i][other_key])
		else:
			val = float(-1)

		result["{}{}{}".format(i, key, other_key)] = val

def avg_value(complete_track, key, other_key, n, result):
	for i in range(0, n):
		if i < len(complete_track["analysis"][key]):
			val = sum(j for j in complete_track["analysis"][key][i][other_key]) / len(complete_track["analysis"][key][i][other_key])
		else:
			val = float(-1)
	
		result["{}{}{}".format(i, key, other_key)] = val

def get_ml_features(
	complete_track,
	feature_set,
	length_dict,
	ska_tracks
):
	result = {}

	if ska_tracks != None:
		if complete_track["id"] in ska_tracks:
			result["ska"] = float(True)
		else:
			result["ska"] = float(False)
	
	for i in feature_set:
		key = i["key"]
		sec = i["sec"]
		n = length_dict[key]
		fun_res = i["fun"](complete_track, key, sec, n, result)
		
	return result

def get_ska_tracks():
	return get_playlist_tracks(
		play_name=SKA_PLAYLIST
	)

def gen_classifier(target_var="", df={}, features=[]):
	x = df[features] # Features
	y = df[target_var] # Target variable

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

	clf = tree.DecisionTreeClassifier(
		max_depth=5
	)
	clf.fit(x_train, y_train)

	y_pred = clf.predict(x_test)

	dot_data = tree.export_graphviz(clf, out_file=None) 
	graph = graphviz.Source(dot_data) 
	graph.render("tree_{}".format(target_var))
	graph = graphviz.Source(dot_data)

	
	return clf, accuracy_score(y_test, y_pred)

def is_it_ska(
	clf=tree.DecisionTreeClassifier(),
	length_dict=[],
	feature_set=[],
	track_id=""
):
	track_complete = {
		"id" : track_id,
		"analysis" : get_analysis_for_track(track_id)
	}

	ml_set = get_ml_features(
		track_complete,
		length_dict,
		feature_set
	)

	cols = list(ml_set.keys())
	df = pd.DataFrame([ml_set], columns=cols)

	x = df[cols]

	return clf.predict_proba(x)

def get_track_data():
	training_tracks = get_playlist_tracks(
		play_name=SOURCE_PLAYLIST
	)

	ska_tracks = get_playlist_tracks(
		play_name=SKA_PLAYLIST
	)

	track_analysis = get_analysis_for_tracks(
		tracks=training_tracks
	)

	print("getting generes from Spotfiy")
	def processTrainingTracks(i):
		return {
			"id" : training_tracks[i],
			"analysis" : track_analysis[i]
		}
	
	tracks_complete = []
	for i in range(0, len(training_tracks)):
		tracks_complete.append(
			processTrainingTracks(i)
		)

	n_beats = int(sum(len(i["analysis"]["beats"]) for i in tracks_complete) / len(tracks_complete))
	n_bars = int(sum(len(i["analysis"]["bars"]) for i in tracks_complete) / len(tracks_complete))
	n_sections = int(sum(len(i["analysis"]["sections"]) for i in tracks_complete) / len(tracks_complete))
	n_segments = int(sum(len(i["analysis"]["segments"]) for i in tracks_complete) / len(tracks_complete))
	n_tatums = int(sum(len(i["analysis"]["tatums"]) for i in tracks_complete) / len(tracks_complete))

	length_dict = {
		"beats" : n_beats,
		"sections" : n_sections,
		"segments" : n_segments,
		"tatums" : n_tatums,
		"bars" : n_bars
	}

	return tracks_complete, ska_tracks, length_dict


def create_clf(
	tracks_complete,
	ska_tracks,
	feature_set,
	length_dict
):
	def process_complete_track(i):
		return get_ml_features(
			i,
			list(feature_set),
			length_dict,
			ska_tracks=ska_tracks
		)

	# ml_set = []
	# for i in tracks_complete:
	# 	ml_set.append(
	# 		process_complete_track(i)
	# 	)
	num_cores = int(multiprocessing.cpu_count())
	ml_set = Parallel(n_jobs=num_cores)(delayed(process_complete_track)(i) for i in tracks_complete)

	cols = list(ml_set[0].keys())
	df = pd.DataFrame(ml_set, columns=cols)
	cols.remove("ska")

	clf, acc = gen_classifier(
		target_var="ska",
		features=cols,
		df=df,
	)

	n_beats = length_dict["beats"]
	n_sections = length_dict["sections"]
	n_segments = length_dict["segments"]
	n_tatums = length_dict["tatums"]

	toDump = ClfInfo()
	toDump.clf = clf
	toDump.n_beats = n_beats
	toDump.n_sections = n_sections
	toDump.n_segments = n_segments
	toDump.n_tatums = n_tatums
	toDump.feature_set = feature_set
	toDump.length_dict = length_dict

	name = 'clf/{}_clf_{}.bin'.format(int(acc * 1000), get_feature_set_hash(feature_set))

	pickle.dump(toDump, open(name, 'wb'), pickle.HIGHEST_PROTOCOL)
	
	return acc

def get_feature_set_hash(feature_set):
	parts = []
	for feature in feature_set:
		parts.append(
			"{}_{}_{}".format(feature["fun"].__name__, feature["key"], feature["sec"])
		)

	name = '_'.join(parts)
	name_hash = hashlib.sha1(name.encode('utf-8')).hexdigest()
	return "{}".format(name_hash)

def recreate_training_playlist():
	ska_tracks = get_ska_tracks()

	tracks = get_playlist_tracks(play_name="training set")
	tracks = [i for i in tracks if i not in ska_tracks]

	random.shuffle(tracks)

	tracks = tracks[:len(ska_tracks)]

	add_tracks_to_playlist(
		playlist_id=get_playlist_by_name("new training set")[0],
		tracks=tracks
	)

	add_tracks_to_playlist(
		playlist_id=get_playlist_by_name("new training set")[0],
		tracks=ska_tracks
	)

def main():
	set_login(
		api_auth=API_AUTH,
		refresh_token=REFRESH_TOKEN,
	)

	feature_sets_complete = [
 		{ "fun" : transform, "key" : "bars", "sec" : "duration"},
 		{ "fun" : transform, "key" : "beats", "sec" : "duration"},
 		{ "fun" : transform, "key" : "tatums", "sec" : "duration"},
 		{ "fun" : transform, "key" : "sections", "sec" : "duration"},
 		{ "fun" : transform, "key" : "sections", "sec" : "tempo"},
 		{ "fun" : transform, "key" : "sections", "sec" : "key"},
 		{ "fun" : transform, "key" : "sections", "sec" : "mode"},
 		{ "fun" : transform, "key" : "sections", "sec" : "loudness"},
 		# { "fun" : transform, "key" : "sections", "sec" : "tempo_confidence"},
 		{ "fun" : transform, "key" : "sections", "sec" : "time_signature"},
 		{ "fun" : transform, "key" : "segments", "sec" : "duration"},
 		{ "fun" : transform, "key" : "segments", "sec" : "loudness_max"},
 		{ "fun" : avg_value, "key" : "segments", "sec" : "pitches"},
 		{ "fun" : avg_value, "key" : "segments", "sec" : "timbre"},
	]

	tracks_complete, ska_tracks, length_dict = get_track_data()

	feature_sets = []
	for i in range(0, len(feature_sets_complete) + 1):
		feature_sets.extend(
			list(itertools.combinations(feature_sets_complete, i))
		)

	random.shuffle(feature_sets)
	
	for feature_set in feature_sets:
		start_time = datetime.utcnow()
		pattern = re.compile(
			"\\d\\d\\d_clf_{}.bin".format(
				get_feature_set_hash(feature_set)
			)
		)

		file_exists = [0 for filename in os.listdir("./clf") if pattern.match(filename)]

		if(
			len(feature_set) > 0 and
			len(file_exists) == 0
		):
			create_clf(
				ska_tracks=ska_tracks,
				tracks_complete=tracks_complete,
				feature_set=feature_set,
				length_dict=length_dict
			)

		print_progress_bar(
			feature_sets.index(feature_set),
			len(feature_sets) + 1,
			prefix="Creating ml sets",
			suffix="took {:.2f} secs".format((datetime.utcnow() - start_time).total_seconds())
		)

	print_progress_bar(feature_sets.index(feature_set), len(feature_sets) + 1, prefix="Creating ml sets")

	return 
	clfInfo = pickle.load(open('clf.bin', 'rb'))
	track_id = "1ccFnIocW7FZ546v5cVqta"

	if track_id in get_ska_tracks():
		raise Exception("Already Learned about this one")

	ska_prob = list(is_it_ska(
		clf=clfInfo.clf,
		length_dict=clfInfo.length_set,
		feature_set=clfInfo.feature_set,
		track_id=track_id,
		n_beats=clfInfo.n_beats,
		n_sections=clfInfo.n_sections,
		n_segments=clfInfo.n_segments,
		n_tatums=clfInfo.n_tatums
	))[0]

	track_info = get_track(track_id)
	print("prob of {} being ska is {:.2f}%".format(track_info["name"], ska_prob[1] * 100))

main()