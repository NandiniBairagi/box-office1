

#!pip3 install spotipy
import spotipy
import sys
from spotipy.oauth2 import SpotifyClientCredentials
import pprint
import webbrowser
from  textblob import TextBlob
import random

from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data
client_id="38123817a7554e81bf633dd282dc2507"
client_secret="517a459b19f14b20a9a30f681cf96003"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API
#name = "{Artist Name}" #chosen artist
result = sp.search("happy") #search query
url=result['tracks']['items'][0]['artists'][0]['external_urls']['spotify']

File_object = open("url.txt","w+")
File_object.write(url)

#Extract Artist's uri
artist_uri = result['tracks']['items'][0]['artists'][0]['uri']

#Pull all of the artist's albums
sp_albums = sp.artist_albums(artist_uri, album_type='album')

#Store artist's albums' names' and uris in separate lists
album_names = []
album_uris = []
for i in range(len(sp_albums['items'])):
    album_names.append(sp_albums['items'][i]['name'])
    album_uris.append(sp_albums['items'][i]['uri'])
       
album_names
album_uris
#album_name

def albumSongs(uri):
    album = uri #assign album uri to a_name
    spotify_albums[album] = {} #Creates dictionary for that specific album
#Create keys-values of empty lists inside nested dictionary for album
    spotify_albums[album]['album'] = [] #create empty list
    spotify_albums[album]['track_number'] = []
    spotify_albums[album]['id'] = []
    spotify_albums[album]['name'] = []
    spotify_albums[album]['uri'] = []
    tracks = sp.album_tracks(album) #pull data on album tracks
    for n in range(len(tracks['items'])): #for each song track
        spotify_albums[album]['album'].append(album_names[album_count]) #append album name tracked via album_count
        spotify_albums[album]['track_number'].append(tracks['items'][n]['track_number'])
        spotify_albums[album]['id'].append(tracks['items'][n]['id'])
        spotify_albums[album]['name'].append(tracks['items'][n]['name'])
        spotify_albums[album]['uri'].append(tracks['items'][n]['uri'])

spotify_albums = {}
album_count = 0
for i in album_uris: #each album
    albumSongs(i)
    print("Album " + str(album_names[album_count]) + " songs has been added to spotify_albums dictionary")
    album_count+=1 #Updates album count once all tracks have been added

def audio_features(album):
    #Add new key-values to store audio features
    spotify_albums[album]['acousticness'] = []
    spotify_albums[album]['danceability'] = []
    spotify_albums[album]['energy'] = []
    spotify_albums[album]['instrumentalness'] = []
    spotify_albums[album]['liveness'] = []
    spotify_albums[album]['loudness'] = []
    spotify_albums[album]['speechiness'] = []
    spotify_albums[album]['tempo'] = []
    spotify_albums[album]['valence'] = []
    spotify_albums[album]['popularity'] = []
    #create a track counter
    track_count = 0
    for track in spotify_albums[album]['uri']:
        #pull audio features per track
        features = sp.audio_features(track)
        
        #Append to relevant key-value
        spotify_albums[album]['acousticness'].append(features[0]['acousticness'])
        spotify_albums[album]['danceability'].append(features[0]['danceability'])
        spotify_albums[album]['energy'].append(features[0]['energy'])
        spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness'])
        spotify_albums[album]['liveness'].append(features[0]['liveness'])
        spotify_albums[album]['loudness'].append(features[0]['loudness'])
        spotify_albums[album]['speechiness'].append(features[0]['speechiness'])
        spotify_albums[album]['tempo'].append(features[0]['tempo'])
        spotify_albums[album]['valence'].append(features[0]['valence'])
        #popularity is stored elsewhere
        pop = sp.track(track)
        spotify_albums[album]['popularity'].append(pop['popularity'])
        track_count+=1

import time
import numpy as np
sleep_min = 2
sleep_max = 5
start_time = time.time()
request_count = 0
for i in spotify_albums:
    audio_features(i)
    request_count+=1
    if request_count % 5 == 0:
        print(str(request_count) + " playlists completed")
        time.sleep(np.random.uniform(sleep_min, sleep_max))
        print('Loop #: {}'.format(request_count))
        print('Elapsed Time: {} seconds'.format(time.time() - start_time))

dic_df = {}
dic_df['album'] = []
dic_df['track_number'] = []
dic_df['id'] = []
dic_df['name'] = []
dic_df['uri'] = []
dic_df['acousticness'] = []
dic_df['danceability'] = []
dic_df['energy'] = []
dic_df['instrumentalness'] = []
dic_df['liveness'] = []
dic_df['loudness'] = []
dic_df['speechiness'] = []
dic_df['tempo'] = []
dic_df['valence'] = []
dic_df['popularity'] = []
for album in spotify_albums: 
    for feature in spotify_albums[album]:
        dic_df[feature].extend(spotify_albums[album][feature])
        
len(dic_df['album'])

import pandas as pd
df = pd.DataFrame.from_dict(dic_df)
df

print(len(df))
final_df = df.sort_values('popularity', ascending=False).drop_duplicates('name').sort_index()
print(len(final_df))

final_df.to_csv("sentiment.txt")

name=df.iloc[0:,3]

name



print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[0]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[1]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[2]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[3]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[4]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[5]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[6]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[7]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[8]+"'>")
print("<meta http-equiv='refresh' content=2; URL='https://www.youtube.com/results?search_query="+name[9]+"'>")
#song_url=song["tracks"][random.randint(1,5)]["artists"][0]['external_urls']["spotify"]
#song_url

