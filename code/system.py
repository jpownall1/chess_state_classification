"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""

from typing import List

import numpy as np
import scipy.linalg

N_DIMENSIONS = 10
CLASS_LABELS = [".", "r", "R", "p", "P", "k", "K", "b", "B", "q", "Q", "n", "N"]


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    Perform nearest neighbour classification.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    #train.shape = 6400, 10 (so each individual image (100 images, 64 places) and its 10 features)
    #test.shape = 1600, 10 (so each individual image (25 images, 64 places) and its 10 features)
    #train_labels.shape = 6400 1D array (each corresponding class label to trains features)
    #label.shape = 1600

    model = process_training_data(train, train_labels)
    train_fvector = np.array(model['fvectors_train'])
    print(train_fvector[1])

    features = np.arange(0, train.shape[1])
    train = train[:, features]
    test = test[:, features]
    n_images = test.shape[0]
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())     # cosine distance
    nearest = np.argmax(dist, axis=1)
    label = train_labels[nearest]
    
    return label


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    #When running evaluate i get the error: 'boolean index did not match indexed array along dimension 0; dimension 
    #is 1600 but corresponding boolean dimension is 6400'
    # When just running train,  it works ok, but when evaluating another part of the dictionary uses this method -
    # the key is "fvectors_train", I dont understand what this is for?
    # print(model.keys())                                #dict_keys(['labels_train', 'fvectors_train'])
    #print(np.array(model['fvectors_train']).shape)      #(6400, 10)
    #print(np.array(model['labels_train']).shape)        #(6400,)
    #print(data.shape)                                   #(1600, 2500)
    #print(np.array(model['fvectors_train']))            #just eigen values (i think?)
    #print(new_data.shape)
    #----------------------------------------------------------------------------------------------------------------------

    if 'fvectors_train' in model:
        return np.array(model['fvectors_train'])
    
    labels = np.array(model['labels_train'])
    d12_pairs = []
    for i in CLASS_LABELS:
        for j in CLASS_LABELS:
            if i != j:
                adata = data[labels == i, :]
                bdata = data[labels == j, :]
                try:
                    d12 = divergence(adata, bdata)
                    d12_pairs.append(d12)
                except:
                    print("Divide by 0 occured")
                

    avg_d12 = np.sum(d12_pairs, axis = 0)
    sorted_indexes = np.argsort(-avg_d12)
    features = sorted_indexes[0:40]

    new_data = data[:,features]
    covx = np.cov(new_data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 10, N - 1))  #gets the last 10 eigenvectors
    v = np.fliplr(v)                                         #v is the eigenvectors
    pca_data = np.dot((new_data - np.mean(new_data)), v)     #this is in the form y=Ax, where v is A and x is the feature vector
    
    return pca_data



def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    
    model = {}
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)

def divergence(class1, class2):
    """compute a vector of 1-D divergences
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    returns: d12 - a vector of 1-D divergence scores
    """
    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12