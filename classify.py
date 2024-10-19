import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Spotify API credentials
client_id = 'bb16ba341c7a412698de2a2c723c86ca'  
client_secret = 'c214bfe80def455688064f82d063f9a9'  

# Authenticate Spotify API
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Function to get artist data from Spotify
def get_artist_data(artist_name):
    result = sp.search(q='artist:' + artist_name, type='artist')
    artist = result['artists']['items'][0]
    return {
        'name': artist['name'],  # Use the exact name returned by Spotify API
        'genres': artist['genres'],
        'popularity': artist['popularity'],
        'followers': artist['followers']['total']
    }

# Function to get a list of related artists for a starting set of artists
def get_related_artists(start_artist_name, num_artists=10):
    artist = sp.search(q='artist:' + start_artist_name, type='artist')['artists']['items'][0]
    artist_id = artist['id']
    related_artists = sp.artist_related_artists(artist_id)['artists']
    
    artist_names = [start_artist_name] + [artist['name'] for artist in related_artists[:num_artists-1]]
    return artist_names

start_artist = "Kanye West"  
artist_names = get_related_artists(start_artist, num_artists=20)  # Fetch 20 related artists

# Collect data for the selected artists
artist_data = []
for artist in artist_names:
    artist_data.append(get_artist_data(artist))

# Create DataFrame
df = pd.DataFrame(artist_data)

all_genres = sorted(list(set([genre for sublist in df['genres'] for genre in sublist])))
for genre in all_genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

# Normalize and clean the data
df['popularity'] = df['popularity'] / 100  # Normalize popularity between 0 and 1
df['followers'] = np.log1p(df['followers'])  # Log-transform followers to handle large numbers

# For this analysis, we use the features: Genres, Popularity, and Followers
features = all_genres + ['popularity', 'followers']

# Calculate the Euclidean distance between artists
distance_matrix = euclidean_distances(df[features])

distance_df = pd.DataFrame(distance_matrix, index=df['name'].str.lower(), columns=df['name'].str.lower())  # Ensure lower case for consistency

def get_similar_artists(query_artist, top_n=10):
    query_artist_lower = query_artist.lower()  # Convert to lower case for matching
    if query_artist_lower not in distance_df.columns:
        print(f"Artist '{query_artist}' not found in dataset!")
        print("Check the available artists:", distance_df.columns)  # Debugging
        return []
    distances = distance_df[query_artist_lower].sort_values()
    return distances.index[1:top_n+1]  # Exclude the first one since it's the same artist

print("Artists in dataset:", distance_df.index)

query_artists = ["Kanye West", "Drake", "Lil Wayne"]

# Output top 10 similar artists for each query artist
for artist in query_artists:
    print(f"Top 10 artists similar to {artist}:")
    print(get_similar_artists(artist, top_n=10))
    print("\n")
