import numpy as np
from joblib import Memory
import progressbar
from joblib import Memory
import pandas as pd

def init(cachedir):
    global memory
    memory = Memory(cachedir=cachedir, verbose=0)

def get_playlist_metadata(spotify_client, playlists):

    for playlist in playlists:

        # get user and playlist_id from uri
        (_,_,user,_,playlist_id) = playlist["uri"].split(":")

        # retrieve playlist metadat from Spotify
        playlist_metadata = spotify_client.user_playlist(user        = user,
                                                         playlist_id = playlist_id)

        # extract relevant information
        playlist["user"]        = user
        playlist["playlist_id"] = playlist_id
        playlist["num_tracks"]  = playlist_metadata["tracks"]["total"]

        # initialize fields for further processing
        playlist["track_ids"]   = []
        
    return playlists


def get_track_ids(sp, playlists):

    # max Spotify batch size
    batch_size = 100

    # retrieve tracks for each playlist
    for playlist in playlists:

        # batch processing
        for offset in np.arange(0, playlist["num_tracks"], batch_size):

            limit = np.min([batch_size, playlist["num_tracks"] - offset])

            playlist_entries = sp.user_playlist_tracks(user        = playlist["user"],
                                                       playlist_id = playlist["playlist_id"], 
                                                       limit       = limit, 
                                                       offset      = offset,
                                                       fields      = ["items"])

            playlist["track_ids"].extend([entry["track"]["id"] for entry in playlist_entries["items"]])
            
    return playlists


def aggregate_metadata(raw_track_data):

    metadata = []

    for playlist_name, spotify_data in raw_track_data:

        track_metadata, album_metadata, artist_metadata, _, _ = spotify_data

        # get year of album release
        release_date = album_metadata["release_date"]

        if album_metadata["release_date_precision"] != "year":
            release_date = release_date.split("-")[0]

        # assamble metadata
        metadata.append([track_metadata["id"],
                         artist_metadata["name"], 
                         track_metadata["name"], 
                         album_metadata["name"],
                         album_metadata["label"],
                         track_metadata["duration_ms"],
                         track_metadata["popularity"],
                         release_date,
                         artist_metadata["genres"], 
                         playlist_name])

    metadata = pd.DataFrame(metadata, columns=["track_id", "artist_name", "title", "album_name", "label", 
                                               "duration", "popularity",  "year",  "genres", "playlist"])
    
    return metadata


def aggregate_features(seq_data, track_data, metadata, with_year=False, with_popularity=False):

    calc_statistical_moments = lambda x: np.concatenate([x.mean(axis=0), x.std(axis=0)])
    
    # sequential data
    segments = seq_data["segments"]
    sl       = len(segments)
    
    # MFCCs - 24 dimensions
    mfcc              = np.array([s["timbre"] for s in segments])
    mfcc              = calc_statistical_moments(mfcc)
    
    # Chroma / pitch classes - 24 dimensions
    chroma            = np.array([s["pitches"] for s in segments])
    chroma            = calc_statistical_moments(chroma)
    
    # maximum loudness values per segment - 2 dimensions
    loudness_max      = np.array([s["loudness_max"] for s in segments]).reshape((sl,1))
    loudness_max      = calc_statistical_moments(loudness_max)
    
    # offset of max loudness value within segment - 2 dimensions
    loudness_start    = np.array([s["loudness_start"] for s in segments]).reshape((sl,1))
    loudness_start    = calc_statistical_moments(loudness_start)
    
    # length of max loudness values within segment - 2 dimensions
    loudness_max_time = np.array([s["loudness_max_time"] for s in segments]).reshape((sl,1))
    loudness_max_time = calc_statistical_moments(loudness_max_time)
    
    # length of segment - 2 dimensions
    duration          = np.array([s["duration"] for s in segments]).reshape((sl,1))
    duration          = calc_statistical_moments(duration)
    
    # confidence of segment boundary detection - 2 dimensions
    confidence        = np.array([s["confidence"] for s in segments]).reshape((sl,1))
    confidence        = calc_statistical_moments(confidence)
    
    # concatenate sequential features
    sequential_features = np.concatenate([mfcc, chroma, loudness_max, loudness_start, 
                                          loudness_max_time, duration, confidence], axis=0)
    
    # track-based data
    track_features = [track_data[0]["acousticness"],     # acoustic or not?
                      track_data[0]["danceability"],     # danceable?
                      track_data[0]["energy"],           # energetic or calm?
                      track_data[0]["instrumentalness"], # is somebody singing?
                      track_data[0]["liveness"],         # live or studio?
                      track_data[0]["speechiness"],      # rap or singing?
                      track_data[0]["tempo"],            # slow or fast?
                      track_data[0]["time_signature"],   # 3/4, 4/4, 6/8, etc.
                      track_data[0]["valence"]]          # happy or sad?
    
    if with_year:
        track_features.append(int(metadata["year"]))
        
    if with_popularity:
        track_features.append(int(metadata["popularity"]))
        
    
    return np.concatenate([sequential_features, track_features], axis=0)


def aggregate_featuredata(raw_track_data, metadata):

    feature_data = []

    for i, (_, spotify_data) in enumerate(raw_track_data):

        _, _, _, f_sequential, f_trackbased = spotify_data

        feature_vec = aggregate_features(f_sequential, 
                                         f_trackbased, 
                                         metadata.iloc[i], 
                                         with_year       = True, 
                                         with_popularity = True)    

        feature_data.append(feature_vec)

    feature_data = np.asarray(feature_data)
    
    return feature_data