# Music-Recommendation-System
## SENG474 - Data Mining Final Project
Below is the report submitted for my final project. I worked on it entirely alone and learned a lot during the process! It has been my favourite project to date!
Note - I recommend reading the jupyter files for more information and visualization of the data.

## Abstract
Music is an essential part of life. It can raise moods, get people excited,
or make them relaxed and calm. It allows us nearly all emotions that we
experience in our lives. Furthermore, the ability to listen to an unlimited
amount of music has never been easier with ‘streaming’ accounting for 83%
of revenue by format in the United States. Moreover, music streaming
users sit around 4.360 billion on the available services.
Therefore, it is necessary to have advanced recommendation systems in
place to help find new music, and retain users. In this work the utiliza-
tion of two clustering methods (MiniBatchKMeans and BIRCH) to find
new music recommendations. The song data is provided from Spotify’s
API, which consists of several continuous audio features. Relevant au-
dio features are transformed by scaling, making them more effective for
use in the clustering algorithms. Once the appropriate parameters were
decided, PCA decomposition was used to visualize cluster results. Sig-
nificant improvements can be achieved by exploring neural networks as
recommendation systems, and using gradient boosted decision trees. Due
to the limitations of the input data, the users’ recommendations will be
muddled.

## 1 Introduction
### 1.1 Music recommendation systems
Music recommendation systems are present in every single streaming service
and provide several playlists that are updated weekly. However, for this project
we have focused on Spotify, and the way they recommend music.
Firstly, “Spotify employs several independent ML models and algorithms to
generate item representations and user representations” [1], by using content-
based and collaborative filtering. Content-based filtering aims to describe the
track by examining the content itself, and collaborative filtering aims to describe
the track within its connection to other tracks on the platform.
When a new song is added to the platform, artist-sourced metadata is ana-
lyzed and is then passed downstream to another analyzer for the raw audio
signals that then become the tracks audio features. These audio features are
the first component of Spotify’s audio analysis system. However, the other audio
analysis features include temporal structure, and NLP models for lyric analy-
sis, web-crawled data, and user-generated playlists. Furthermore, Spotify has
“widely publicized collaborative filtering as the driving power behind its recom-
mendation engine” [1]. Thus, when user A enjoys songs X, Y, and Z, and user
B enjoys X and Y, but has not listened to Z yet, Z is then recommended to user
B. Luckily for Spotify, because of this they have access to a massive user-item
interaction matrix covering all users and tracks. However, issues have arised
with “...accuracy, scalability, speed, and cold start problems.”
Lastly, to wrap up the simplistic view of Spotify’s recommendation system, the
goal and rewards of the system are user retention, time spent on the platform,
and satisfaction. However, these goals are too broad for a balanced reward
system. So success is relative to where and why a user engages with the system.
For example, with the users’ discover weekly playlist, this week’s data, historical
data, and this week’s cluster data are fed into a gradient boosted decision tree

### 1.2 Problem definition and related work
The main goal is to build a music recommendation system based on clustering
methods given a user’s playlist. The user’s playlist is provided a link to it, and
then each song in the playlist has its audio features extracted which are the
main components for clustering.
Recommendation systems are increasingly important as we continue to indulge
in so much accessibility to music, movies, tv shows, live streaming and more.
Hence, recommendation systems are a fundamental piece of AI.
Specifically, the importance of music recommendation was recognized at AIcrowd,
where the Spotify Million Playlist Dataset Challenge took place. It was an open-
ended challenge for music recommendation research where several thousands
participated.

### 1.3 Dataset
Input data was obtained two separate ways. Firstly, the music catalog that
served as the database was obtained from Kaggle ([2]) and is one of the largest
Spotify datasets out there. Next, the user provided datasets were obtained from
a link to the playlist, and then each playlist was iterated through to get all the
necessary audio features from each track. Table 1 describes all the features of
the input data.
#### Table 1: Dataset features
| Name | Description | Value |
| :------: | :-----------: | :------: |
| id | Spotify ID for the track | string |
| name | name of the track | string | 
| popularity | popularity of the artist | integer | 
| duration_ms | duration of the track in milliseconds | integer | 
| artists  | name of the artist(s) | array of objects | 
| id_artists | Spotify ID(s) of the artist(s) | array of objects | 
| release_date | Date the album was first released | string | 
| danceability | description of how suitable a track is for dancing based on musical elements | number<float> | 
| energy | measure from 0.0 to 1.0 representing a perceptual measure of intensity and activity | number<float> |
| key | the key the track is in | integer | 
| loudness | overall loudness of the track in decibels | number<float> | 
| mode | indicates the major or minor of a track, either 1 or 0 | integer | 
| speechiness | the presence of words spoken in a track | number<float> |
| acousticness | the presence of words spoken in a track | number<float> | 
| instrumentalness | predicts whether a track contains no vocals | number<float> | 
| liveness | presence of an audience in the recording | number<float> | 
| valence | measure from 0.0 to 1.0 to describing the musical positiveness in a track | number<float> | 
| tempo | estimate of beats per minute | number<float> | 
| time_signature | estimation of how many beats are in each bar | integer | 

 
## 2 Approach
Below, two clustering methods used to tackle the problem are explained. Before,
we could cluster the data and make recommendations, a track catalog of 600,000
songs was downloaded from Kaggle, and then Spotify’s API was used to get the
user’s tracks. Followed by scaling of the appropriate columns that would be
used in clustering.

### 2.1 Getting user playlist track information
User’s have to get the link to their playlist from either the Spotify web app,
desktop app, or mobile app. Once that is achieved, the developer must verify
his API tokens before communicating with Spotify. Next, each playlist was
iterated through to get the audio features of the tracks, along with some of the
artist and album information. It was done to produce the same order/style of
the dataset downloaded from Kaggle. Once, all the information from the playlist
was pulled from Spotify’s API, it was stored into a DataFrame and then saved
locally.

### 2.2 MinMaxScaling
Before running the clustering algorithms, MinMaxScaling was applied to the
catalog of tracks, as well as the user’s playlists. MinMaxScaling is a method
for scaling numeric data by linear transformation. It scales the data such that
the minimum value in the data becomes 0, and the maximum value becomes
1. This is useful for data that has a wide range of values and is not normally
distributed, as it can help to normalize the data and make it more amenable to
certain types of modeling.

### 2.3 MiniBatchKMeans
The first clustering algorithm applied was MiniBatchKMeans which is a variant
of the k-means clustering algorithm, which is a popular method for grouping
data into clusters of similar items. In the standard k-means algorithm, the entire
dataset is used in each iteration to update the cluster centers. In contrast, Mini-
BatchKMeans uses a smaller subset of the data to update the cluster centers in
each iteration. This can make the algorithm faster and more scalable, particu-
larly for large datasets. Moreover, k-means++ initialization was used as it’s a
more sophisticated method for selecting initial centroids. This can help avoid
suboptimal solutions and improve the overall performance of the algorithm.

### 2.4 BIRCH
Next, the second clustering algorithm applied was BIRCH (Balanced Iterative
Reducing and Clustering using Hierarchies) which is a clustering algorithm for
grouping a set of data points into clusters. It is a hierarchical clustering algo-
rithm that uses a tree-based data structure to store the data points and their
clusters. BIRCH is efficient and scalable, and can handle large datasets with
high dimensionality. It is particularly useful for outlier detection and for dealing
with noisy or uncertain data.

### 2.5 Silhouette scoring
When finding the optimal number of clusters, silhouette scoring was used.
Where a silhouette score is a measure of how well-defined a cluster of data
points is in a dataset. It is calculated by measuring the average distance be-
tween each data point in a cluster and all other data points in the same cluster,
and then taking the mean of those distances for all the data points in the cluster.
The silhouette score ranges from -1 to 1, with a high value indicating that the
data points in the cluster are well-separated from each other and a low value in-
dicating that the data points are densely packed together or overlap with other
clusters.

## 3 Results <To Improve> 
When doing MiniBatchKMeans clustering, the whole catalog was used to train
the model with a different number of clusters. Through the silhouette scoring
used, the optimal number of clusters was 8, and so the final model was fitted
with 8 clusters. It produced an extremely dense result as Figure 2 shows the
1D scatter plot of the clustering after PCA decomposition with 3 components.

Next, the results of the BIRCH clustering visually look significantly different.
However, the catalog was this time randomly sampled to 200,000 items to signif-
icantly reduce the amount of time required to complete computation. Silhouette
scoring was again used to get the optimal number of clusters which was 5. Figure
2 below shows the result of another 3 dimensional PCA decomposition

## 4 Conclusion <To Improve>
Unfortunately, the accuracy of these clustering models is quite subjective. How-
ever, some of the unique songs recommended were awesome! Further improve-
ments can definitely be made, specifically, having larger samples of user’s playlists
would be a great help, and perhaps being able to eliminate genres not in the
playlists would also work. Furthermore, trying to implement a gradient boosted
decision tree would also be interesting, as well as seeing what a neural net
recommendation system could achieve!

## References
[1] “How Spotify’s Algorithm Works? A Complete Guide to Spotify Recommenda-
tion System [2022]: Music Tomorrow Blog.” How Spotify’s Algorithm Works? A
Complete Guide to Spotify Recommendation System [2022] | Music Tomorrow Blog,
https://www.music-tomorrow.com/blog/how-spotify-recommendation-system-works-a-
complete-guide-2022.
[2] “Spotify Million Playlist Dataset Challenge: Challenges.” AIcrowd, https://www.aicrowd.com/challenges/spotify-
million-playlist-dataset-challenge.
Pedregosa, F., Varoquaux, Ga"el, Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
. . . others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine
Learning Research, 12(Oct), 2825–2830.
