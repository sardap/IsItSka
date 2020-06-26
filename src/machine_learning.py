import copy
import os
import multiprocessing
import graphviz
import pickle
import io
import itertools
import hashlib
import re
import random
import redis
import json

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score
from sklearn import metrics, tree
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from datetime import datetime
from multiprocessing import Process, JoinableQueue
from threading import Semaphore, Thread

from env import API_AUTH, REFRESH_TOKEN, SOURCE_PLAYLIST, SKA_PLAYLIST, CLF_FOLDER_PATH
from spotify_helper import get_analysis_for_tracks, get_access_token, get_playlist_tracks, get_genres_for_track, set_login, get_analysis_for_track, get_track, add_tracks_to_playlist, get_playlist_by_name, remove_tracks_from_playlist


os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"

MIN_CONFIDENC1E = 0.7

FEATURES = [
	"beats"
]

_loaded_clfs = []

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

_cache = {}

def get_feature_key(fun, key, other_key, n, id):
	return "{}{}{}{}".format(key, other_key, n, id)

def transform(complete_track, key, other_key, n, redis_connection=None):
	feature_key = get_feature_key("transform", key, other_key, n, complete_track["id"])
	if redis_connection != None:
		x = redis_connection.get(feature_key)
		if x != None:
			return json.loads(x)
	# if feature_key in _cache:
	# 	return _cache[feature_key]

	result = {}
	for i in range(0, n):
		if i < len(complete_track["analysis"][key]):
			val = float(complete_track["analysis"][key][i][other_key])
		else:
			val = float(-1)

		result["{}{}{}".format(i, key, other_key)] = val

	# _cache[feature_key] = result
	if redis_connection != None:
		redis_connection.set(feature_key, json.dumps(result))
		
	return result

def avg_value(complete_track, key, other_key, n, redis_connection=None):
	feature_key = get_feature_key("avg_value", key, other_key, n, complete_track["id"])
	if redis_connection != None:
		x = redis_connection.get(feature_key)
		if x != None:
			return json.loads(x)
	# if feature_key in _cache:
	# 	return _cache[feature_key]

	result = {}

	for i in range(0, n):
		if i < len(complete_track["analysis"][key]):
			val = sum(j for j in complete_track["analysis"][key][i][other_key]) / len(complete_track["analysis"][key][i][other_key])
		else:
			val = float(-1)
	
		result["{}{}{}".format(i, key, other_key)] = val

	if redis_connection != None:
		redis_connection.set(feature_key, json.dumps(result))
	# _cache[feature_key] = result
	return result

def get_ml_features(
	complete_track,
	feature_set,
	length_dict,
	ska_tracks=None,
	redis_connection=None,
):
	# pipeline = redis_connection.pipeline()
	# all_keys = []
	# for i in feature_set:
	# 	key = i["key"]
	# 	sec = i["sec"]
	# 	n = length_dict[key]
	# 	feature_key = get_feature_key(
	# 		i["fun"].__name__,
	# 		key,
	# 		sec,
	# 		n,
	# 		complete_track["id"]
	# 	)
	# 	pipeline.get(feature_key)
	# 	all_keys.append(feature_key)

	# for h in pipeline.execute():
	# 	if h != None:
	# 		result = json.loads(h)
	# 		for key in result.keys():
	# 			result[key] = result[key]
	result = {}

	for i in feature_set:
		key = i["key"]
		sec = i["sec"]
		n = length_dict[key]
		fun_res = i["fun"](complete_track, key, sec, n, redis_connection)

		for key in fun_res.keys():
			result[key] = fun_res[key]

	if ska_tracks != None:
		if complete_track["id"] in ska_tracks:
			result["ska"] = float(True)
		else:
			result["ska"] = float(False)
		
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
		# max_depth=20,
		# min_samples_split=5,
		# max_features="log2"
	)
	clf.fit(x_train, y_train)

	y_pred = clf.predict(x_test)

	# dot_data = tree.export_graphviz(clf, out_file=None) 
	# graph = graphviz.Source(dot_data) 
	# graph.render("tree_{}".format(target_var))
	# graph = graphviz.Source(dot_data)

	
	return clf, accuracy_score(y_test, y_pred)

def _is_it_ska(
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
		complete_track=track_complete,
		feature_set=feature_set,
		length_dict=length_dict
	)

	cols = list(ml_set.keys())
	df = pd.DataFrame([ml_set], columns=cols)

	x = df[cols]

	return clf.predict_proba(x)

def ska_prob(track_id):
	prop_sum = 0

	clfInfo = _loaded_clfs[0]
	ska_prob = list(_is_it_ska(
		clf=clfInfo.clf,
		length_dict=clfInfo.length_dict,
		feature_set=clfInfo.feature_set,
		track_id=track_id
	))[0]
	return ska_prob[1]

	prop_sum += ska_prob[1]

	return prop_sum / len(_loaded_clfs)

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
	length_dict,
	redis_connection=None,
	ml_set=None
):
	if ml_set == None:
		def process_complete_track(i):
			return get_ml_features(
				i,
				list(feature_set),
				length_dict,
				ska_tracks=ska_tracks,
				redis_connection=redis_connection
			)

		ml_set = []
		for i in tracks_complete:
			ml_set.append(
				process_complete_track(i)
			)
		# num_cores = int(multiprocessing.cpu_count())
		# ml_set = Parallel(n_jobs=num_cores)(delayed(process_complete_track)(i) for i in tracks_complete)

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

	dump = pickle.dumps(toDump)
	
	return dump, name, acc

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
	new_training_set_id = get_playlist_by_name("new training set")[0]

	remove_tracks_from_playlist(
		playlist_id=new_training_set_id,
		tracks=get_playlist_tracks(
			play_id=new_training_set_id
		)
	)

	ska_tracks = get_ska_tracks()

	tracks = get_playlist_tracks(play_name="not ska training")
	tracks = [i for i in tracks if i not in ska_tracks]

	# random.shuffle(tracks)

	tracks = tracks[:len(ska_tracks)]


	add_tracks_to_playlist(
		playlist_id=new_training_set_id,
		tracks=tracks
	)

	add_tracks_to_playlist(
		playlist_id=new_training_set_id,
		tracks=ska_tracks
	)

def progress_bar_ml_create(progress, n, acc, delta, delta_avg):
	print_progress_bar(
		progress,
		n,
		prefix="Creating ml sets",
		suffix="acc {:.2f} took {:.2f} secs avg {:.2f}".format(acc, delta, delta_avg)
	)

def create_clf_worker(
	length_dict,
	ska_tracks,
	tracks_complete,
	work_queue,
	work_sem,
	saver_work_queue,
	saver_work_sem
):
	redis_connection = redis.Redis(
		host='localhost',
		port=6379,
		db=0
	)

	while True:
		work_sem.acquire()
		top = work_queue.pop(0)
		if top == None:
			saver_work_queue.append(None)
			saver_work_sem.release()
			break

		feature_set = top["feature_set"]

		dump, name, acc = create_clf(
			ska_tracks=ska_tracks,
			tracks_complete=tracks_complete,
			feature_set=feature_set,
			length_dict=length_dict,
			redis_connection=redis_connection
		)

		saver_work_queue.append({
			"name" : name,
			"dump" : dump,
			"acc" : acc
		})
		saver_work_sem.release()

def clf_saver(total_features, work_queue, work_sem):
	count = 0
	delta_histroy = []
	while True:
		start_time = datetime.utcnow()
		work_sem.acquire()
		top = work_queue.pop(0)
		if top == None:
			break

		with open(top["name"], "wb") as f:
			f.write(top["dump"])
			
		count += 1
		delta = (datetime.utcnow() - start_time).total_seconds()
		delta_histroy.append(delta)
		progress_bar_ml_create(
			count,
			total_features,
			top["acc"],
			delta,
			sum(delta_histroy) / len(delta_histroy)
		)

def create_ml_complete_set(
	complete_tracks,
	ska_tracks,
	length_dict,
	complete_feature_set
):
	result = {}

	for complete_track in complete_tracks:
		track_feature = {}

		if ska_tracks != None:
			if complete_track["id"] in ska_tracks:
				track_feature["ska"] = float(True)
			else:
				track_feature["ska"] = float(False)
		
		for feature in complete_feature_set:
			key = feature["key"]
			sec = feature["sec"]
			n = length_dict[key]
			feature_key = get_feature_key(
				fun=feature["fun"].__name__,
				key=key,
				other_key=sec,
				n=n,
				id=complete_track["id"]
			)
			track_feature[feature_key] = feature["fun"](complete_track, key, sec, n)

		result[complete_track["id"]] = track_feature
		
	return result


def create_all_clf():
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
 		{ "fun" : transform, "key" : "segments", "sec" : "loudness_max_time"},
 		{ "fun" : avg_value, "key" : "segments", "sec" : "pitches"},
 		{ "fun" : avg_value, "key" : "segments", "sec" : "timbre"},
	]

	tracks_complete, ska_tracks, length_dict = get_track_data()

	# print("making complete ml set")
	# create_ml_complete_set(
	# 	complete_tracks=tracks_complete,
	# 	ska_tracks=ska_tracks,
	# 	length_dict=length_dict,
	# 	complete_feature_set=feature_sets_complete
	# )

	print("making Getting all feature set combos")
	feature_sets = []
	for i in range(0, len(feature_sets_complete) + 1):
		feature_sets.extend(
			list(itertools.combinations(feature_sets_complete, i))
		)

	print("Removing already completed entries")
	files = os.listdir("./clf")

	def process(feature_set, files):
		pattern = re.compile(
			"\\d\\d\\d_clf_{}.bin".format(
				get_feature_set_hash(feature_set)
			)
		)

		file_exists = [0 for filename in files if pattern.match(filename)]

		if len(feature_set) == 0 or len(file_exists) > 0:
			return feature_set

		return None
		
	num_cores = int(multiprocessing.cpu_count())
	to_remove = Parallel(n_jobs=num_cores)(delayed(process)(i, files) for i in feature_sets)

	for i in to_remove:
		if i != None:
			feature_sets.remove(i)

	# print("creating saver worker")
	# saver_work_queue = []
	# saver_work_sem = Semaphore(0)
	# saver_worker = Thread(
	# 	target=clf_saver,
	# 	args=(
	# 		len(feature_sets), 
	# 		saver_work_queue, 
	# 		saver_work_sem,
	# 	)
	# )
	# saver_worker.start()

	# tracks_complete_ids = [i["id"] for i in tracks_complete]
	# ska_tracks_ids = [i["id"] for i in ska_tracks]

	# print("creating clf workers")
	# clf_work_queue = []
	# clf_work_sem = Semaphore(0)
	# clf_workers = []
	# num_workers = 1
	# for i in range(0, num_workers):
	# 	clf_worker = Thread(
	# 		target=create_clf_worker,
	# 		args=(
	# 			length_dict,
	# 			copy.deepcopy(ska_tracks_ids),
	# 			copy.deepcopy(tracks_complete_ids),
	# 			clf_work_queue,
	# 			clf_work_sem,
	# 			saver_work_queue,
	# 			saver_work_sem,
	# 		)
	# 	)

	# 	clf_worker.start()
	# 	clf_workers.append(clf_worker)

	# print("Creating work queue")
	# for feature_set in feature_sets:
	# 	clf_work_queue.append({
	# 		"feature_set" : feature_set
	# 	})
	# 	clf_work_sem.release()

	# for i in range(0, num_workers):
	# 	clf_work_queue.append(None)
	# 	clf_work_sem.release()

	# print("Waiting for everyone to finish")
	# saver_worker.join()

	# for worker in clf_workers:
	# 	worker.join()

	# def process(feature_set, ml_complete_set, q):
	# 	ml_set = []

	# 	for track in tracks_complete:
	# 		entry = {}
	# 		track_features_complete = ml_complete_set[track["id"]]

	# 		entry["ska"] = track_features_complete["ska"]
	# 		for feature in feature_set:
	# 			feature_key = get_feature_key(
	# 				fun=feature["fun"].__name__,
	# 				key=feature["key"],
	# 				other_key=feature["sec"],
	# 				n=length_dict[feature["key"]],
	# 				id=track["id"]
	# 			)
	# 			for k, v in track_features_complete[feature_key].items():
	# 				entry[k] = v

	# 		ml_set.append(entry)

	# 	create_clf(
	# 		ska_tracks=ska_tracks,
	# 		tracks_complete=tracks_complete,
	# 		feature_set=feature_set,
	# 		length_dict=length_dict,
	# 		ml_set=ml_set,
	# 		queue=q
	# 	)

	# # for feature_set in feature_sets:
	# # 	process(feature_set, ml_complete_set)
	# q = JoinableQueue()
	# p = Process(target=saver, args=(q,))
	# p.start()
	# print("Starting clf gennration ")
	# num_cores = int(multiprocessing.cpu_count())
	# Parallel(n_jobs=num_cores)(delayed(process)(i, ml_complete_set, q) for i in feature_sets)
	# q.put(None) # Poison pill
	# q.join()
	# p.join()
	# return

	redis_connection = redis.Redis(
		host='localhost',
		port=6379,
		db=0
	)

	delta_histroy = []
	for feature_set in feature_sets:
		start_time = datetime.utcnow()
		acc = 0.0

		if(len(feature_set) > 0):
			dump, name, acc = create_clf(
				ska_tracks=ska_tracks,
				tracks_complete=tracks_complete,
				feature_set=feature_set,
				length_dict=length_dict,
				redis_connection=redis_connection
			)

			with open(name, "wb") as f:
				f.write(dump)

		delta = (datetime.utcnow() - start_time).total_seconds()
		delta_histroy.append(delta)
		if len(delta_histroy) > 0:
			delta_avg = sum(delta_histroy) / len(delta_histroy)

		progress_bar_ml_create(
			feature_sets.index(feature_set),
			len(feature_sets),
			acc,
			delta,
			delta_avg
		)

	print_progress_bar(len(feature_sets), len(feature_sets) + 1, prefix="Creating ml sets")

def get_top_x_clf(x, percent=False):
	ratings = []

	target_floder = CLF_FOLDER_PATH

	for filename in os.listdir(target_floder):
		ratings.append({
			"rating" : int(filename[:3]),
			"filename" : "{}/{}".format(target_floder, filename)
		})

	ratings.sort(key=lambda x: x["rating"], reverse=True)

	if percent:
		length = int(len(ratings) * x)
	else:
		length = x
	return ratings[:min(length, len(ratings))]

def load_clfs():
	clfs_filenames = get_top_x_clf(5)

	for to_load in clfs_filenames:
		_loaded_clfs.append(
			pickle.load(open(to_load["filename"], 'rb'))
		)

def create_fresh_clf():
	# recreate_training_playlist()

	# create_all_clf()

	clf_to_test = get_top_x_clf(0.1, percent=True)

	tracks_complete, ska_tracks, length_dict = get_track_data()

	for i in clf_to_test:
		clfInfo = pickle.load(open(i["filename"], 'rb'))
		track_id = "6UNf12sZXLcvQUbNpyfKHd"

		if track_id in ska_tracks:
			raise Exception("Already Learned about this one")

		ska_prob = list(_is_it_ska(
			clf=clfInfo.clf,
			length_dict=clfInfo.length_dict,
			feature_set=clfInfo.feature_set,
			track_id=track_id
		))[0]

		# acc, dump = create_clf(
		# 	tracks_complete=tracks_complete,
		# 	ska_tracks=ska_tracks,
		# 	feature_set=clfInfo.feature_set,
		# 	length_dict=clfInfo.length_dict,
		# 	redis_connection=None
		# )

		# print("New acc {}".format(acc))

		track_info = get_track(track_id)
		print("prob of {} being ska is {:.2f}%".format(track_info["name"], ska_prob[1] * 100))