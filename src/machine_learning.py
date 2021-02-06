import struct
import time
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
import gzip
import spotipy

import pandas as pd
import numpy as np

from io import StringIO  # Python 3.x
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics, tree, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from datetime import datetime
from multiprocessing import Process, JoinableQueue
from threading import Semaphore, Thread

from env import API_AUTH, SOURCE_PLAYLIST, SKA_PLAYLIST, CLF_FOLDER_PATH, REDIS_ML_DB, REDIS_IP, REDIS_PORT, NOT_SKA_PLAYLIST
from spotify_helper import get_playlist_tracks, get_features_for_tracks


FEATURES = [
    "beats"
]

_loaded_clfs = []


class ClfInfo():
    clf = None
    feature_set = None


def get_feature_key(fun, key, other_key, n, id):
    return "{}:{}:{}:{}:{}".format(fun, key, other_key, n, id)


def compress_string(data):
    return gzip.compress(data.encode("utf-8"))


def decompress_to_string(data):
    return gzip.decompress(data).decode("utf-8")


def transform(complete_track, key, other_key, n, redis_connection=None):
    feature_key = get_feature_key(
        transform.__name__, key, other_key, n, complete_track["id"])

    result = {}
    for i in range(0, n):
        if i < len(complete_track["analysis"][key]):
            val = float(complete_track["analysis"][key][i][other_key])
        else:
            val = float(-1)

        result["{}{}{}".format(i, key, other_key)] = val

    return result


def avg_value(complete_track, key, other_key, n, redis_connection=None):
    feature_key = get_feature_key(
        avg_value.__name__, key, other_key, n, complete_track["id"])

    result = {}

    for i in range(0, n):
        if i < len(complete_track["analysis"][key]):
            val = sum(j for j in complete_track["analysis"][key][i][other_key]) / len(
                complete_track["analysis"][key][i][other_key])
        else:
            val = float(-1)

        result["{}{}{}".format(i, key, other_key)] = val

    return result


def string_to_bytes(complete_track, key, other_key, n, redis_connection=None):

    feature_key = get_feature_key(
        string_to_bytes.__name__, key,
        other_key, n, complete_track["id"])

    result = {}

    for i in range(0, n):
        if i < len(complete_track["analysis"][key][other_key]):
            x = complete_track["analysis"][key][other_key][i]
            val = float(ord(x))
        else:
            val = -1

        result["{}{}{}".format(i, key, other_key)] = val

    return result


def get_ml_features(
    complete_track,
    feature_set,
    ska_tracks,
):
    result = {}

    for key in feature_set:
        result[key] = int(complete_track['features'][key] * 1000)

    if ska_tracks != None:
        if complete_track["id"] in ska_tracks:
            result["ska"] = float(True)
        else:
            result["ska"] = float(False)

    return result


def gen_descion_tree_clf():
    return tree.DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=3,
        max_features="log2"
    )


def gen_random_forest_clf():
    return RandomForestClassifier(
        max_depth=None,
        n_estimators=1000,
        min_samples_leaf=3,
        random_state=None
    )


def gen_nearest_neighbors_clf():
    return NearestNeighbors(
        n_neighbors=5,
        algorithm='ball_tree'
    )


def gen_neural_network_clf():
    return MLPClassifier(
        solver='adam',
        hidden_layer_sizes=(100, 50,),
        random_state=1,
        max_iter=50000
    )


def gen_classifier(clf_gen, target_var, df, features):
    x = df[features]  # Features
    y = df[target_var]  # Target variable

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)  # 70% training and 30% test

    clf = clf_gen()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    return clf, accuracy_score(y_test, y_pred)


def _is_it_ska(
        clf=tree.DecisionTreeClassifier(),
        length_dict=[],
        feature_set=[],
        track_id=""
):
    track_complete = {
        "id": track_id,
        "analysis": get_analysis_for_track(track_id)
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

    while len(_loaded_clfs) == 0:
        time.sleep(0.5)

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
        playlist_id=SOURCE_PLAYLIST
    )

    ska_tracks = get_playlist_tracks(
        playlist_id=SKA_PLAYLIST
    )

    track_features = get_features_for_tracks(
        tracks=training_tracks
    )

    def processTrainingTracks(i):
        return {
            "id": training_tracks[i],
            "features": track_features[i]
        }

    tracks_complete = []
    for i in range(0, len(training_tracks)):
        tracks_complete.append(
            processTrainingTracks(i)
        )

    return tracks_complete, ska_tracks


def get_ml_pipeline(
        complete_track,
        feature_set,
        length_dict,
        ska_tracks,
        pipeline,
        redis_connection
):
    for i in feature_set:
        key = i["key"]
        sec = i["sec"]
        if key in length_dict:
            n = length_dict[key]
        elif sec in length_dict:
            n = length_dict[sec]
        else:
            raise Exception("missing entry in legnth dict")

        feature_key = get_feature_key(
            i["fun"].__name__,
            key,
            sec,
            n,
            complete_track["id"]
        )

        pipeline.get(feature_key)


def create_clf(
    tracks_complete,
    ska_tracks,
    feature_set,
):
    ml_set = [get_ml_features(
        complete_track=i,
        feature_set=feature_set,
        ska_tracks=ska_tracks,
    ) for i in tracks_complete]

    cols = list(ml_set[0].keys())
    df = pd.DataFrame(ml_set, columns=cols)
    cols.remove("ska")

    clf, acc = gen_classifier(
        clf_gen=gen_random_forest_clf,
        target_var="ska",
        features=cols,
        df=df,
    )

    toDump = ClfInfo()
    toDump.clf = clf
    toDump.feature_set = feature_set

    name = 'clf/{}_clf_{}.bin'.format(
        int(acc * 1000),
        '_'.join(feature_set))

    dump = pickle.dumps(toDump)

    return dump, name, acc


def create_all_clf():
    tracks_complete, ska_tracks = get_track_data()

    tracks_complete_dump = json.dumps(tracks_complete)
    ska_tracks_dump = json.dumps(ska_tracks)

    delta_histroy = []
    feature_set = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    dump, name, acc = create_clf(
        ska_tracks=ska_tracks,
        tracks_complete=tracks_complete,
        feature_set=feature_set,
    )

    print("created new CLF with acc:{}".format(acc))

    with open(name, "wb") as f:
        f.write(dump)


def get_top_x_clf(x, percent=False):
    ratings = []

    target_floder = CLF_FOLDER_PATH

    for filename in os.listdir(target_floder):
        ratings.append({
            "rating": int(filename[:3]),
            "filename": "{}/{}".format(target_floder, filename)
        })

    ratings.sort(key=lambda x: x["rating"], reverse=True)

    if percent:
        length = int(len(ratings) * x)
    else:
        length = x
    return ratings[:min(length, len(ratings))]


def load_clfs():

    _loaded_clfs.clear()

    clfs_filenames = get_top_x_clf(5, percent=False)

    for to_load in clfs_filenames:
        _loaded_clfs.append(
            pickle.load(open(to_load["filename"], 'rb'))
        )

    if len(_loaded_clfs) == 0:
        raise Exception("No Clf loaded")


def recreate_best_clf():
    result = False

    tracks_complete, ska_tracks = get_track_data()

    clfs_filenames = get_top_x_clf(1, percent=True)

    for clfInfo in _loaded_clfs:
        dump, name, acc = create_clf(
            tracks_complete=tracks_complete,
            ska_tracks=ska_tracks,
            feature_set=clfInfo.feature_set,
            length_dict=clfInfo.length_dict,
            use_pipeline=False
        )

        file_info = clfs_filenames[_loaded_clfs.index(clfInfo)]
        old_acc = file_info["rating"]

        print("Acc:{} old Acc:{}".format(acc * 1000, old_acc))

        if acc * 1000 > old_acc + 1:
            os.remove(file_info["filename"])

            with open(name, "wb") as f:
                f.write(dump)

            print("new model is better than old model")
            result = True

    return result


def create_fresh_clf():

    create_all_clf()
