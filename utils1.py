import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import collections
import torch
import pandas as pd
import tensorflow as tf
import utilss
from functools import partial
import numpy as np
from scipy.sparse import csr_matrix
import math


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def getSamplingAdj1(adjList, sampling_idx_range): #eachRow, the total sampling idx
    newNodeAdjList = []
    sampling_idx_range = sampling_idx_range
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            # newNodeAdjList.append(sampling_idx_range.index(listIndex))
            newNodeAdjList.append(listIndex)

    return newNodeAdjList


def getSamplingAdj(adjList, sampling_idx_range): #eachRow, the total sampling idx
    newNodeAdjList = []
    sampling_idx_range = sampling_idx_range
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            newNodeAdjList.append(sampling_idx_range.index(listIndex))
            # newNodeAdjList.append(listIndex)

    return newNodeAdjList

def samplingFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, name):
    trainsl = samplingTrainsetLabel.tolist()
    label = [trainsl[i].index(1) for i in range(len(trainsl))]   #get val index/class

    label0Index = [i for i, x in enumerate(label) if x == 0]
    samplingIndex0 = random.sample(range(0, len(label0Index)), sampleNumEachClass)
    samplingFixedIndex = np.array(label0Index)[samplingIndex0].tolist()

    label1Index = [i for i, x in enumerate(label) if x == 1]
    samplingIndex1 = random.sample(range(0, len(label1Index)), sampleNumEachClass)
    samplingFixedIndex = samplingFixedIndex + np.array(label1Index)[samplingIndex1].tolist()

    label2Index = [i for i, x in enumerate(label) if x == 2]
    samplingIndex2 = random.sample(range(0, len(label2Index)), sampleNumEachClass)
    samplingFixedIndex = samplingFixedIndex + np.array(label2Index)[samplingIndex2].tolist()

    label3Index = [i for i, x in enumerate(label) if x == 3]
    samplingIndex3 = random.sample(range(0, len(label3Index)), sampleNumEachClass)
    samplingFixedIndex = samplingFixedIndex + np.array(label3Index)[samplingIndex3].tolist()

    label4Index = [i for i, x in enumerate(label) if x == 4]
    samplingIndex4 = random.sample(range(0, len(label4Index)), sampleNumEachClass)
    samplingFixedIndex = samplingFixedIndex + np.array(label4Index)[samplingIndex4].tolist()

    label5Index = [i for i, x in enumerate(label) if x == 5]
    samplingIndex5 = random.sample(range(0, len(label5Index)), sampleNumEachClass)
    samplingFixedIndex = samplingFixedIndex + np.array(label5Index)[samplingIndex5].tolist()

    label6Index = [i for i, x in enumerate(label) if x == 6]
    samplingIndex6 = random.sample(range(0, len(label6Index)), sampleNumEachClass)
    samplingFixedIndex = samplingFixedIndex + np.array(label6Index)[samplingIndex6].tolist()   #get the sampling train index

    name = 'TrainLabelIndex' + name
    dataframe = pd.DataFrame({'TrainLabelIndex0': samplingIndex0, 'TrainLabelIndex1': samplingIndex1,
                              'TrainLabelIndex2': samplingIndex2, 'TrainLabelIndex3': samplingIndex3,
                              'TrainLabelIndex4': samplingIndex4, 'TrainLabelIndex5': samplingIndex5,
                              'TrainLabelIndex6': samplingIndex6})  # save the samplingIndex of every client
    dataframe.to_csv(name, index=False, sep=',')  #

    return samplingFixedIndex


def loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, name):   #这边要的是索引
    name = 'TrainLabelIndex' + name
    nd = np.genfromtxt(name, delimiter=',', skip_header=True)
    samplingIndex = np.array(nd).astype(int)

    samplingIndex0 = samplingIndex[:, 0]
    samplingIndex1 = samplingIndex[:, 1]
    samplingIndex2 = samplingIndex[:, 2]
    samplingIndex3 = samplingIndex[:, 3]
    samplingIndex4 = samplingIndex[:, 4]
    samplingIndex5 = samplingIndex[:, 5]
    samplingIndex6 = samplingIndex[:, 6]

    samplingFixedIndex = samplingIndex0.tolist()
    samplingFixedIndex += samplingIndex1.tolist()
    samplingFixedIndex += samplingIndex2.tolist()
    samplingFixedIndex += samplingIndex3.tolist()
    samplingFixedIndex += samplingIndex4.tolist()
    samplingFixedIndex += samplingIndex5.tolist()
    samplingFixedIndex += samplingIndex6.tolist()

    # trainsl = samplingTrainsetLabel.tolist()
    # label = [trainsl[i].index(1) for i in range(len(trainsl))]   #get val index/class
    #
    # label0Index = [i for i, x in enumerate(label) if x == 0]
    # samplingIndex0 = random.sample(range(0, len(label0Index)), sampleNumEachClass)
    # samplingFixedIndex = np.array(label0Index)[samplingIndex0].tolist()
    #
    # label1Index = [i for i, x in enumerate(label) if x == 1]
    # samplingIndex1 = random.sample(range(0, len(label1Index)), sampleNumEachClass)
    # samplingFixedIndex = samplingFixedIndex + np.array(label1Index)[samplingIndex1].tolist()
    #
    # label2Index = [i for i, x in enumerate(label) if x == 2]
    # samplingIndex2 = random.sample(range(0, len(label2Index)), sampleNumEachClass)
    # samplingFixedIndex = samplingFixedIndex + np.array(label2Index)[samplingIndex2].tolist()
    #
    # label3Index = [i for i, x in enumerate(label) if x == 3]
    # samplingIndex3 = random.sample(range(0, len(label3Index)), sampleNumEachClass)
    # samplingFixedIndex = samplingFixedIndex + np.array(label3Index)[samplingIndex3].tolist()
    #
    # label4Index = [i for i, x in enumerate(label) if x == 4]
    # samplingIndex4 = random.sample(range(0, len(label4Index)), sampleNumEachClass)
    # samplingFixedIndex = samplingFixedIndex + np.array(label4Index)[samplingIndex4].tolist()
    #
    # label5Index = [i for i, x in enumerate(label) if x == 5]
    # samplingIndex5 = random.sample(range(0, len(label5Index)), sampleNumEachClass)
    # samplingFixedIndex = samplingFixedIndex + np.array(label5Index)[samplingIndex5].tolist()
    #
    # label6Index = [i for i, x in enumerate(label) if x == 6]
    # samplingIndex6 = random.sample(range(0, len(label6Index)), sampleNumEachClass)
    # samplingFixedIndex = samplingFixedIndex + np.array(label6Index)[samplingIndex6].tolist()   #get the sampling train index
    #
    # name = 'TrainLabelIndex' + name
    # dataframe = pd.DataFrame({'TrainLabelIndex0': samplingIndex0, 'TrainLabelIndex1': samplingIndex1,
    #                           'TrainLabelIndex2': samplingIndex2, 'TrainLabelIndex3': samplingIndex3,
    #                           'TrainLabelIndex4': samplingIndex4, 'TrainLabelIndex5': samplingIndex5,
    #                           'TrainLabelIndex6': samplingIndex6})  # save the samplingIndex of every client
    # dataframe.to_csv(name, index=False, sep=',')  #

    return samplingFixedIndex

def getSamplingGlobalAdj(graph, sampling_idx_range):   #graph:the whole graph, sampling_idx_range:the whole sampling idx
    adjLen = len(graph)
    samplingGlobalAdj = collections.defaultdict(list)
    for idx in range(adjLen):
        withInFlag = idx in sampling_idx_range
        if withInFlag:
            currentList = graph[idx]
            newCurrentList = getSamplingAdj1(currentList, sampling_idx_range)
            samplingGlobalAdj[idx] = newCurrentList
        else:
            samplingGlobalAdj[idx] = []

    samplingMatrix = getSamplingMatrix(samplingGlobalAdj, sampling_idx_range)

    return samplingMatrix

def getSamplingMatrix(samplingGlobalAdjWithReduceNode, sampling_idx_range):
    adjLen = len(samplingGlobalAdjWithReduceNode)
    samplingMatrix = np.zeros((adjLen, adjLen))

    for idx in sampling_idx_range:
        currentList = samplingGlobalAdjWithReduceNode[idx]
        for listIdx in currentList:
            samplingMatrix[idx, listIdx] = 1

    return samplingMatrix

def samplingData(graph, labels, trainLabel, samplingRate, sampleNumEachClass, name):
    testNum = 1000
    valNum = 500
    totalSampleNum = len(graph) - testNum - valNum    #get the train num according to the fixed testset and valset
    samplingNum = int(samplingRate * totalSampleNum)  #get sampling trainset num according to the sampling rate

    nd = np.genfromtxt(name, delimiter=',', skip_header=True)
    samplingIndex = np.array(nd).astype(int).tolist()

    # if name == 'client3.csv':
    #     nd = np.genfromtxt(name, delimiter=',', skip_header=True)
    #     samplingIndex = np.array(nd).astype(int).tolist()
    #
    #     # file_handle = open(name)  # read data from .csv
    #     # data = pd.read_csv(file_handle, index_col=0)
    #     # samplingIndex = data.tolist()
    #
    # else:
    # samplingIndex = random.sample(range(0, totalSampleNum), samplingNum)  # get random samplingIndex
    # dataframe = pd.DataFrame({'samplingIndex': samplingIndex})  # save the samplingIndex of every client
    # dataframe.to_csv(name, index=False, sep=',')  #

    train_sampling_idx_range = np.sort(samplingIndex)                           #sort the sampling index
    sampling_idx_range = train_sampling_idx_range.tolist() + [i for i in range(totalSampleNum, len(graph))]   #get all data index of sampling adj
    samplingTrainsetLabel = trainLabel[train_sampling_idx_range]      #the new graph train set label
    samplingLabels = labels[sampling_idx_range]

    samplingFixedIndex = loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, name) # the sampling fixed class of the new graph train set

    samplingMatrix = getSamplingGlobalAdj(graph, sampling_idx_range)  #get the global adj matrix with reduced node

    count = 0
    samplingAdj = collections.defaultdict(list)
    for index in sampling_idx_range:
        currentList = graph[index]
        newCurrentList = getSamplingAdj(currentList, sampling_idx_range)

        samplingAdj[count] = newCurrentList
        count += 1

    return samplingAdj, samplingNum, samplingFixedIndex, sampling_idx_range, samplingLabels, samplingMatrix

def getTheCutSampling(samplingAdj, name):
    # name = 'cut' + name
    # adjLen = len(samplingAdj)
    # samplingNum = int(adjLen * 0.5)   #cut 30% nodes from train set
    # samplingCutIndexs = random.sample(range(0, adjLen), samplingNum)
    # dataframe = pd.DataFrame({'samplingCutIndex': samplingCutIndexs})  # save the samplingIndex of every client
    # dataframe.to_csv(name, index=False, sep=',')  #

    name = 'cut' + name

    nd = np.genfromtxt(name, delimiter=',', skip_header=True)
    samplingCutIndexs = np.array(nd).astype(int).tolist()
    #
    for cutIdx0 in samplingCutIndexs:
        cutRow0 = samplingAdj[cutIdx0]
        if(cutRow0 != []):
            cutIdx1 = cutRow0[len(cutRow0) - 1]
            cutPos0 = len(cutRow0) - 1
            cutRow0.pop(cutPos0)   #remove the last element

            cutRow1 = samplingAdj[cutIdx1]
            cutPos1 = cutRow1.index(cutIdx0)
            cutRow1.pop(cutPos1)

            samplingAdj[cutIdx0] = cutRow0
            samplingAdj[cutIdx1] = cutRow1

    return samplingAdj

def load_data_amazon_photo(dataset_str):
    data = np.load(dataset_str)

    adj_data = data['adj_data']
    adj_indices = data['adj_indices']
    adj_indptr = data['adj_indptr']
    adj_shape = data['adj_shape']
    attr_data = data['attr_data']
    attr_indices = data['attr_indices']
    attr_indptr = data['attr_indptr']
    attr_shape = data['attr_shape']
    labels = data['labels']
    class_names = data['class_names'] # none useness

    labelsCode = tf.keras.utils.to_categorical(labels)  #one_hot code
    adj = csr_matrix((adj_data, adj_indices, adj_indptr), adj_shape)  #convert to csr_matrix
    features = csr_matrix((attr_data, attr_indices, attr_indptr), attr_shape).tolil()

    idx_test = range(2800, features.shape[0]) # get the last 2800 indexes as test set
    idx_val = range(1400, 2800) # get 1400 indexes as val set
    idx_train = samplingFixedTraindata(labelsCode[0:1400, :], 60, 'test') # sampling 420 indexes as train set, each class has 60 labels

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labelsCode.shape)
    y_val = np.zeros(labelsCode.shape)
    y_test = np.zeros(labelsCode.shape)
    y_train[train_mask, :] = labelsCode[train_mask, :]
    y_val[val_mask, :] = labelsCode[val_mask, :]
    y_test[test_mask, :] = labelsCode[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data_amazon_photo1(dataset_str, samplingRate, sampleNumEachClass, cutFlag, name):
    data = np.load(dataset_str)

    adj_data = data['adj_data']
    adj_indices = data['adj_indices']
    adj_indptr = data['adj_indptr']
    adj_shape = data['adj_shape']
    attr_data = data['attr_data']
    attr_indices = data['attr_indices']
    attr_indptr = data['attr_indptr']
    attr_shape = data['attr_shape']
    labels = data['labels']
    class_names = data['class_names'] # none useness

    labelsCode = tf.keras.utils.to_categorical(labels)  #one_hot code
    adj = csr_matrix((adj_data, adj_indices, adj_indptr), adj_shape)  #convert to csr_matrix
    features = csr_matrix((attr_data, attr_indices, attr_indptr), attr_shape).tolil()

    graph = collections.defaultdict(list)
    count = 0
    for i in range(features.shape[0]):
        graph[count] = features.rows[i]
        count += 1

    testSetNum = 2800
    valSetNum = 1400

    allyLen = len(graph) - testSetNum - valSetNum
    ally = labelsCode[0:allyLen, :]
    samplingAdj, samplingNum, samplingTrainFixedIndex, sampling_idx_range, \
    samplingLabels, samplingGlobalAdjWithReduceNode = samplingData(graph, labelsCode, ally, samplingRate,
                                                                   sampleNumEachClass, name)

    if cutFlag:
        samplingAdj = getTheCutSampling(samplingAdj, name)

    samplingFeatures = features[sampling_idx_range, :]
    samplingAdj = nx.adjacency_matrix(nx.from_dict_of_lists(samplingAdj))


    idx_test = range(samplingAdj.shape[0] - testSetNum, samplingAdj.shape[0]) # get the last 2800 indexes as test set
    idx_train = range(samplingAdj.shape[0] - testSetNum - valSetNum, samplingAdj.shape[0] - testSetNum)  # sampling 420 indexes as train set, each class has 60 labels
    idx_val = samplingTrainFixedIndex  # get 1400 indexes as val set

    train_mask = sample_mask(idx_train, samplingLabels.shape[0])
    val_mask = sample_mask(idx_val, samplingLabels.shape[0])
    test_mask = sample_mask(idx_test, samplingLabels.shape[0])

    y_train = np.zeros(samplingLabels.shape)
    y_val = np.zeros(samplingLabels.shape)
    y_test = np.zeros(samplingLabels.shape)
    y_train[train_mask, :] = samplingLabels[train_mask, :]
    y_val[val_mask, :] = samplingLabels[val_mask, :]
    y_test[test_mask, :] = samplingLabels[test_mask, :]

    return samplingAdj, samplingFeatures, y_train, y_val, y_test, \
           train_mask, val_mask, test_mask, samplingGlobalAdjWithReduceNode, sampling_idx_range

def load_data1(dataset_str, samplingRate, sampleNumEachClass, cutFlag, name):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    #allx is the training set, tx is the test set, ally is the label of allx, ty is the label of tx
    x, y, tx, ty, allx, ally, graph = tuple(objects)

    #----------------------modify---------------------------------------#
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))   #get the test set index
    test_idx_range = np.sort(test_idx_reorder)                                          #sort the index of the test set

    features = sp.vstack((allx, tx)).tolil()  # all data
    features[test_idx_reorder, :] = features[test_idx_range, :]  # test features， 为了让特征向量和邻接矩阵的索引一致，把乱序的特征数据读取出来，按正确的ID顺序重新排列。
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))  # all label
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # test label
    #--------------------------------------------------------------------------#
    samplingAdj, samplingNum, samplingTrainFixedIndex, sampling_idx_range, \
    samplingLabels, samplingGlobalAdjWithReduceNode = samplingData(graph, labels, ally, samplingRate, sampleNumEachClass, name)

    if cutFlag:
        samplingAdj = getTheCutSampling(samplingAdj, name)

    # features = sp.vstack((allx, tx)).tolil()  # all data
    samplingFeatures = features[sampling_idx_range, :]
    samplingAdj = nx.adjacency_matrix(nx.from_dict_of_lists(samplingAdj))
    testSetNum = 1000
    valSetNum = 500

    idx_test = range(samplingAdj.shape[0] - testSetNum, samplingAdj.shape[0])  # test set 1000
    idx_train = samplingTrainFixedIndex  # train idx 0-140
    idx_val = range(samplingAdj.shape[0] - testSetNum - valSetNum, samplingAdj.shape[0] - testSetNum)  # valization 500

    train_mask = sample_mask(idx_train, samplingLabels.shape[0])
    val_mask = sample_mask(idx_val, samplingLabels.shape[0])
    test_mask = sample_mask(idx_test, samplingLabels.shape[0])

    y_train = np.zeros(samplingLabels.shape)
    y_val = np.zeros(samplingLabels.shape)
    y_test = np.zeros(samplingLabels.shape)
    y_train[train_mask, :] = samplingLabels[train_mask, :]
    y_val[val_mask, :] = samplingLabels[val_mask, :]
    y_test[test_mask, :] = samplingLabels[test_mask, :]

    return samplingAdj, samplingFeatures, features, y_train, y_val, y_test, \
           train_mask, val_mask, test_mask, samplingGlobalAdjWithReduceNode, sampling_idx_range
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # test_idx_range = np.sort(test_idx_reorder)
    #
    # if dataset_str == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended
    #
    # features = sp.vstack((allx, tx)).tolil()  #all data
    # features[test_idx_reorder, :] = features[test_idx_range, :] #test features
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #
    # labels = np.vstack((ally, ty))  #all label
    # labels[test_idx_reorder, :] = labels[test_idx_range, :] #test label
    #
    # idx_test = test_idx_range.tolist() # test idx
    #
    # idx_train = range(len(y)) # train idx 0-140
    # idx_val = range(len(y), len(y)+500) #valization 140-640
    #
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])
    #
    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(dataset_str):  #modify the load process
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    labels = ally.tolist()
    label = [labels[i].index(1) for i in range(len(labels))]

    label0Index = [i for i, x in enumerate(label) if x == 0]
    label1Index = [i for i, x in enumerate(label) if x == 1]
    label2Index = [i for i, x in enumerate(label) if x == 2]
    label3Index = [i for i, x in enumerate(label) if x == 3]
    label4Index = [i for i, x in enumerate(label) if x == 4]
    label5Index = [i for i, x in enumerate(label) if x == 5]
    label6Index = [i for i, x in enumerate(label) if x == 6]



    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()  #all data
    features[test_idx_reorder, :] = features[test_idx_range, :] #test features， 为了让特征向量和邻接矩阵的索引一致，把乱序的特征数据读取出来，按正确的ID顺序重新排列。
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))  #all label
    labels[test_idx_reorder, :] = labels[test_idx_range, :] #test label

    idx_test = test_idx_range.tolist() # test idx
    idx_train = range(len(y)) # train idx 0-140
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.    #用于检查数字是否为无穷大
    r_mat_inv = sp.diags(r_inv)    #对矩阵对角化
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def   normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj) #创建一个稀疏矩阵
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  #D^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # check the data is inf
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj0(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized.toarray()

def afterprocess_adj1(adj_normalized):
    # shape = adj_normalized.shape
    adjcoo = sp.coo_matrix(adj_normalized)
    # adjcoo = sp.coo_matrix((adj_normalized, shape), shape=(shape[0], shape[1])) #
    return sparse_to_tuple(adjcoo)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders, globalWeight0=None, lobalWeight0=None, wlabels=None, cosine=None):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    # feed_dict.update({placeholders['local_weight']: lobalWeight0})
    # feed_dict.update({placeholders['global_weight']: globalWeight0})
    # feed_dict.update({placeholders['wlabels']: wlabels})
    # feed_dict.update({placeholders['cosine']: cosine})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.shape[0]) + int(target.shape[0])
    # total = tf.concat([source, target], dim=0)
    total = tf.concat([source, target], axis = 0)
    total0 = total.expand_dims()
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i)
                    for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                    for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    # batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    source = torch.from_numpy(source)
    target = torch.from_numpy(target)

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

def guassian_kernel_tf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i)
                    for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                    for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf_tf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    # batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    # source = torch.from_numpy(source)
    # target = torch.from_numpy(target)

    batch_size = int(source.shape[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

def maximum_mean_discrepancy(x, y, kernel=utilss.gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost

def mmd_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the MMD between two representations.

  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      utilss.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight
  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, loss_value)
    tf.losses.add_loss(loss_value)

  print('loss:', loss_value)

  return loss_value

def getGaussianSigma(eps, delta):
    sigma = math.sqrt((2 * math.log(1.25 / delta)) / (eps ** 2))

    return sigma


def getNoiseForPartialGradient(grad, importPos, eps, delta, sensitivity):
    grad = np.array(grad)
    size = grad.shape
    sigma = getGaussianSigma(eps / 200, delta)

    maxCount = np.sum(importPos)

    noise = torch.normal(0, sigma * sensitivity / 2708, size=[maxCount, 1])  #
    count = 0

    for i in range(size[0]):
        for j in range(size[1]):
            if importPos[i][j] == 1:
                grad[i][j] += noise[count]
                count += 1



def getAllNoiseGradient(grad, eps, delta, sensitivity):
    sigma = getGaussianSigma(eps/200, delta)
    size = grad.shape

    for i in range(size[1]):

        noise = torch.normal(0, sigma * sensitivity/2708, size=[size[0], 1])  #

        for j in range(size[0]):
            grad[j][i] += noise[j]

    return grad

def scale(intermediate_layer_output):

    for input_id in range(intermediate_layer_output.shape[0]):
            currentIdOutput = intermediate_layer_output[input_id]
            min = intermediate_layer_output[input_id].min()
            max = intermediate_layer_output[input_id].max()

            if min == 0 and max == 0:
                intermediate_layer_output[input_id] = 0
            else:
                intermediate_layer_output[input_id] = (currentIdOutput - min) / (max - min)

def getSampleConverage(scaleOutVal, threshold):
    samplesNum = scaleOutVal.shape[0]
    neuronNum = scaleOutVal.shape[1]

    result = np.zeros(shape=(samplesNum, neuronNum), dtype=np.uint8)

    for inputId in range(samplesNum):
        for neuronId in range(neuronNum):
            try:
                if scaleOutVal[inputId][neuronId] > threshold:
                    result[inputId][neuronId] = 1
            except:
                print("val:", scaleOutVal[inputId][neuronId])

    return result


def getNeuronConverageFrequency(samplesCoverage):
    num_input, num_neuron = samplesCoverage.shape
    neuronCoverage = np.sum(samplesCoverage, axis=0) / num_input

    return neuronCoverage


def getWeightCoverageFrequency(inputCoverage, outputCoverage):
    feature_num = len(inputCoverage)
    neuron_num = len(outputCoverage)

    weightCoverageFrequency = np.zeros(shape=(feature_num, neuron_num), dtype=np.float)
    all_weight_coverage = []

    for i in range(feature_num):
        for j in range(neuron_num):
            weightCoverageFrequency[i][j] = inputCoverage[i] + outputCoverage[j]
            all_weight_coverage.append((weightCoverageFrequency[i][j], (i, j)))

    return weightCoverageFrequency, all_weight_coverage

def getDenseMatrix(sparseMatrix):
        shape = sparseMatrix[2]
        pos = sparseMatrix[0]
        val = sparseMatrix[1]

        result = np.zeros(shape=(shape[0], shape[1]), dtype=np.float)
        for i in range(pos.shape[0]):
            position = pos[i]
            row = position[0]
            col = position[1]
            result[row][col] = val[i]

        return result


def getImportWeight(weightCoverage, weight, threshold0, threshold1):  # need to check
    weightRow = weight.shape[0]
    weightCol = weight.shape[1]
    weightRec = []

    for i in range(weightRow):
        for j in range(weightCol):
            weightRec.append((weight[i][j], (i, j)))

    sorted_weight_coverage = sorted(weightCoverage, key=lambda item: item[0])
    sorted_weight = sorted(weightRec, key=lambda item: item[0])

    sortedWeightCoverageMin = 0
    sortedWeightCoverageMax = 0
    sortedWeightMin = 0
    sortedWeightMax = 0

    leftWC = []
    leftW = []

    weightCoverageLeft = np.zeros(shape=(weightRow, weightCol), dtype=np.int)
    weightLeft = np.zeros(shape=(weightRow, weightCol), dtype=np.int)

    for eachWC in sorted_weight_coverage:
        if eachWC[0] > threshold0:
            leftWC.append(eachWC)
            row = eachWC[1][0]
            col = eachWC[1][1]
            weightCoverageLeft[row][col] = 1
        if eachWC[0] < sortedWeightCoverageMin:
            sortedWeightCoverageMin = eachWC[0]
        if eachWC[0] > sortedWeightCoverageMax:
            sortedWeightCoverageMax = eachWC[0]

    for eachW in sorted_weight:
        if eachW[0] > threshold1:
            leftW.append(eachW)
            row = eachW[1][0]
            col = eachW[1][1]
            weightLeft[row][col] = 1
        if eachW[0] < sortedWeightMin:
            sortedWeightMin = eachW[0]
        if eachW[0] > sortedWeightMax:
            sortedWeightMax = eachW[0]

    importPos = np.zeros(shape=(weightRow, weightCol), dtype=np.int)
    for i in range(weightRow):
        for j in range(weightCol):
            if weightCoverageLeft[i][j] == 1 and weightLeft[i][j] == 1:
                importPos[i][j] = 1

    return importPos
