import spotipy
import editdistance

from spotipy.oauth2 import SpotifyClientCredentials
from env import CLIENT_ID, CLIENT_SECRET

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET))


def get_spotipy_client():
    return sp


def find_track(track_name, artist_name=None):
    result = get_spotipy_client().search(
        q=track_name,
        type="track", limit=20)

    if(artist_name != None):
        best = 1000
        best_track = None
        for i in range(len(result['tracks']['items'])):
            track = result['tracks']['items'][i]
            lowest = min(
                editdistance.distance(artist_name.lower(), art['name'].lower())
                for art in track['artists'])

            if lowest == 0:
                best_track = track
                break

            if lowest + i < best:
                best = lowest + i
                best_track = track

        return best_track

    return result['tracks']['items'][0]


def get_playlist_tracks(playlist_id):

    tracks = []

    tracksRes = sp.playlist(playlist_id)
    tracksRes = tracksRes['tracks']
    while True:
        for i in tracksRes['items']:
            tracks.append(i['track'])

        if tracksRes['next'] == None:
            break

        tracksRes = sp.next(tracksRes)

    return tracks


def get_features_for_tracks(tracks):
    result = []

    n = 100
    for i in [tracks[i:i + n] for i in range(0, len(tracks), n)]:
        for features in sp.audio_features([j['id'] for j in i]):
            result.append(features)

    return result


def add_track_to_playlist(playlist_id, track_id):
    pass


def remove_track_from_playlist(playlist_id, track_id):
    pass
