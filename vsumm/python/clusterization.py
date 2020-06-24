import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
from math import sqrt

def dist(hist_a, hist_b):
    hist_a = np.array(hist_a)
    hist_b = np.array(hist_b)
    return np.linalg.norm(hist_a-hist_b)

def estimate_k(features):
    threshold = 0.4
    k = 0
    for i in range(len(features)-1):
        if dist(features[i].hist,features[i+1].hist) > threshold:
            k+=1
    return k

def has_converged(centroids, old_centroids):
    return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in old_centroids]))

def cluster_points(features, centroids):
    clusters = {}
    for feat in features:
        distances = [dist(feat.histF,c) for c in centroids]
        nearest = distances.index(min(distances))
        try:
            clusters[nearest].append(feat)
        except KeyError:
            clusters[nearest] = [feat]
    return clusters

def evaluate_centroids(centroids, clusters):
    clusters_hists = {}
    for key, values in clusters.items():
        clusters_hists[key] = []
        for v in values:
            clusters_hists[key].append(v.histF)

    new_centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        new_centroids.append(np.mean(clusters_hists[k], axis = 0))
    return new_centroids

def find_clusters(features):
    k = estimate_k(features)
    old_centroids = [f.histF for f in random.sample(features,k)]
    #centroids = [f.hist for f in random.sample(features,k)]
    centroids = [features[i].histF for i in range(k)]
    clusters = {}
    while not has_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = cluster_points(features,centroids)
        centroids = evaluate_centroids(old_centroids, clusters)

        #for key, values in clusters.items():
        #    print '\nkey', key
        #    for v in values:
        #        print v.frame_id,

    keyframes = find_keyframes(clusters, centroids)
    final_keyframes = remove_similar_keyframes(keyframes)
    return final_keyframes
    
def find_keyframes(clusters, centroids):
    keyframes = []
    for key, values in clusters.items():
        distances = [dist(values[i].histF, centroids[key]) for i in range(len(values))]
        nearest = distances.index(min(distances))
        keyframes.append(values[nearest])
    return keyframes

def remove_similar_keyframes(keyframes):
    keyframes.sort(key=lambda x: x.frame_id)
    index_to_remove = []
    for i in range(len(keyframes)-1):
        if i in index_to_remove:
            continue
        for j in range(i+1, len(keyframes)):
            if j in index_to_remove:
                continue
            if dist(keyframes[i].hist,keyframes[j].hist) < 0.3:
                index_to_remove.append(j)
            
    keyframes = [keyframes[i] for i in range(len(keyframes)) if i not in index_to_remove]
    return keyframes






