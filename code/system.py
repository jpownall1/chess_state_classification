"""Chess piece classification system for printed images of chess boards.

COM2004/3004 assignment

The code in this file is used by both train.py and evaluate.py
It implements the key system functionality, including the dimensionality
reduction and classification steps needed for classifyng the chess board.

author: Jordan Pownall, 190143099
version: v1.0
"""

from typing import List

import numpy as np
import scipy.linalg

N_DIMENSIONS = 10
CLASS_LABELS = [".", "r", "R", "p", "P", "k", "K", "b", "B", "q", "Q", "n", "N"]


def classify(train: np.ndarray, labels_train: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    Perform nearest neighbour classification.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    to_return = []

    for test_sample in test:
        to_return.append(my_KNN(train, labels_train, test_sample))
    
    return to_return


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    #Using the PCA and learning the A matrix to transform to 10 dimensions
    v = np.array(model['eigen_vectors'])

    # compute the mean vector
    datamean = np.mean(data)

    # subtract mean from all data points
    centered = data - datamean

    # project points onto PCA axes
    pca_data = np.dot(centered, v)
    
    return pca_data



def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    This method processes the training data and stores the results in a dictionary. The results
    include the principal components of the training data, the training deature vectors themselves
    and the feature vectors corresponding labels.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["eigen_vectors"] = myPCAEV(fvectors_train, 10).tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        fvectors (np.ndarray): An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    This method uses the classify method on an array of feature vectors using the trest data.
    It does this by using the NN algorithm in the classify function, and passes in the training and test
    feature vectors and the test datas corresponding labels. The training labels and feature vectors 
    are stored in the model.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order'

    This method removes the pawns from the training data to find the next nearest neighbour for
    pawns that are classified on rows 1 and 8 (as pawns cannot be here) and modifies the classify_squares to
    reflect this change.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        to_return (list[str]): A list of one-character strings representing the labels for each square.
    """

    #This part removes the pawns from the training data
    train_data = np.array(model['fvectors_train'])
    train_labels = np.array(model['labels_train'])
    no_pawns_data = train_data[train_labels != "p", :]
    no_pawns_labels = np.delete(train_labels, np.where(train_labels == "p"))
    no_pawns_data = no_pawns_data[no_pawns_labels != "P", :]
    no_pawns_labels = np.delete(no_pawns_labels, np.where(no_pawns_labels == "P"))
    
    #This section checks if a pawn is found on the first or last row, and if so, changes it to its second nearest neighbour
    boards = split_to_boards(classify_squares(fvectors_test, model))
    boardNum = 0
    for board in boards:
        for i, v in enumerate(board):                             #i.e. for index, value
            if (i in [0,1,2,3,4,5,6,7,56,57,58,59,60,61,62,63]):
                if v == "p":
                    board[i] = my_KNN(no_pawns_data, no_pawns_labels, fvectors_test[boardNum*64 + i, :])[0]
                elif v == "P":
                    board[i] = my_KNN(no_pawns_data, no_pawns_labels, fvectors_test[boardNum*64 + i, :])[0]
        boardNum = boardNum + 1
    
    # Adds all fo the lists back into being a single list
    to_return = [label for board in boards for label in board]

    return np.array(to_return)

def myPCAEV(data, n) -> np.ndarray:
    """Apply PCA to reduce dimensionality of data matrix to n dimensions.

    Computes and returns the principal components (eigenvectors of the dataset as column 
    vectors) of the training data using a training dataset of feature vectors.

    Args:
        data (np.ndarray) = the data to be 
        n (int) = the number of dimensions to reduce down to

    Returns:
        v (np.ndarray): the eigen values 
    """
    # compute data covariance matrix
    covx = np.cov(data, rowvar=0)
    # compute first N pca axes
    n_orig = covx.shape[0]
    [d, v] = scipy.linalg.eigh(covx, eigvals=(n_orig - n, n_orig - 1))
    v = np.fliplr(v)

    return v

def split_to_boards(data: List[str]) -> List[str]:
    """Splits a large array of squares into boards for modification and testing.

    This method splits a large array of labels into equal lists of 64 to represent
    different boards.

    Args:
        data (List[str]) = list of ordered square labels

    Returns:
        chunks (List[str]) = A list of lists representing each boards squares
    """
    chunks = [data[x:x+64] for x in range(0, len(data), 64)]
    
    return chunks

def my_KNN(fvectors_train, labels_train, test_sample) -> str:
    """Classifies a single test sample against a training dataset.
    
    Classifies a test sample using k nearest neighbour classification with
    a training dataset and its corresponding labels.
    It does this by finding the 10 nearest neighbours and then out of them finds
    the most frequent label value.

    Args:
        fvectors_train (np.ndarray): 2-D array storing the training feature vectors.
        labels_train (np.ndarray): 1-D array storing the training labels.
        test_sample (np.ndarray): 1-D array storing the test feature vector.

    Returns:
        label (str) = A label produced from classifying the test vector.
    """
    #records all distances from test sample to each training sample
    distances = np.linalg.norm(fvectors_train - test_sample, axis=1)
        
    # Sorts the distances from low to high to get the top k distances
    distance_indexes_sorted = np.argsort(distances)

    # Extracting top 10 neighbours index numbers
    indexes = distance_indexes_sorted[:10]

    # Getting the labels
    k_labels = []
    for i in indexes:
        k_labels.append(labels_train[i])
    
    # Finding the most frequent label in k neighbours
    elements,i = np.unique(k_labels,return_inverse=True)    # finds only different elements and their positions
    counts = np.bincount(i)                                 # count each element
    index = counts.argmax()                                 # finds the index of the most common element
    label = elements[index]                                 # the most common label

        
    return label

"""The following methods are my old methods used in the classification step. My report details why I chose to change from using these methods."""
######################################################################################################################################################################################################
def euclidean_distance(sample1, sample2) -> float:
    """Finds the distance between two vectors.
    
    Calculates the euclidean distance between two vectors.

    Args:
        sample1 (np.ndarray): 1-D feature vector
        sample2 (np.ndarray): 1-D feature vector

    Returns:
        distance (float) = A float representing the distance.
    """
    # This part finds the sum of the squared difference between each feature between two samples
    squared_distance = 0.0
    for i in range(len(sample1)-1):
        squared_distance += (sample1[i] - sample2[i])**2
	
    # This is then square routed to find the euclidean distance
    distance = np.sqrt(squared_distance)

    return distance


def my_NN(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classifies a dataset.
    
    Classifies a test dataset using nearest neighbour classification with
    a training dataset and its corresponding labels.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        label (List[str]) = A list of labels produced from classifying the sample 
        vectors.
    """
    if  (len(test.shape)==1):  # test only has one dimension
     test = np.expand_dims(test, axis=0)   

    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())     # cosine distance
    nearest = np.argmax(dist, axis=1)
    label = train_labels[nearest]
    
    return label
