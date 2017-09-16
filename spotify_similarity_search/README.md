# Finding Similar Songs on Spotify

(c) Alexander Schindler, AIT Austrian Institute of Technology, Technical University of Vienna - [http://ifs.tuwien.ac.at/~schindler](http://ifs.tuwien.ac.at/~schindler)

In this tutorial I will demonstrate how to apply different machine learning techniques to search for similar songs on Spotify. Because the tutorial grew in size, I decided to split it into three parts. 

## Overview

*Part 1* demonstrates the traditional machine learning approach of extracting features from the audio content, training a classifier and predicting results. The tutorial uses the Spotify API to retrieve pre-extracted low-level audio features and meta-data. These features are aggregated into single vector representations, normalized and used to calculate similarities between the tracks. Different approaches are presented to optimize and evaluate this approach. 

*Part 2* is based on the same data as Part 1, but uses Deep Learning models to learn the similarity function. A Siamese architecture with fully connected layers is trained on the aggregated feature vectors as well as bi-directional LSTMs on sequential features. 

*Part 3* introduces methods to estimate similarities between songs based on provide genre labels. These similarities are then used to train the Siamese Networks presented in Part 2.

## Requirements

It uses Python 2.7 as the programming language with the popular Keras and Theano Deep Learning libraries underneath.


Spotipy

TODO: describe how to get client credentials