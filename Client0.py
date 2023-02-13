from __future__ import division
from __future__ import print_function

from flask import *
import flask_socketio
from flask_socketio import *
from random import randrange
import tensorflow as tf
from .utils1 import *
from .models import GCN, MLP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import threading
import torch
import sys

class GcnMode:
    def __init__(self):
        self.sess = tf.Session()
        self.trainLoss = []
        self.trainAcc = []
        self.valLoss = []
        self.valAcc = []

    def load_cora(self, FLAGS):  # load cora dataset
        adj, features, allFeatures, y_train, y_val, y_test, train_mask, \
        val_mask, test_mask, samplingGlobalAdjWithReduceNode, sampling_idx_range = load_data1(FLAGS.dataset, 0.3, 20, True, 'client0.csv')

        return adj, features, allFeatures, y_train, y_val, y_test, train_mask, \
               val_mask, test_mask, samplingGlobalAdjWithReduceNode, sampling_idx_range


    def init_default_parameter(self):   #set parameters
        # Set random seed
        seed = 123
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Settings
        flags = tf.app.flags
        FLAGS = flags.FLAGS

        # flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
        flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
        flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
        flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
        flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
        flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
        flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
        flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

        return FLAGS

    def pre_process(self, FLAGS, features, adj): #some preprocess
        # Some preprocessing
        features = preprocess_features(features)
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN
        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            num_supports = 1
            model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        return num_supports, support, features, model_func

    def modeBuild(self, num_supports, features, y_train, model_func, FLAGS):  #build train model
        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        }

        # Create model
        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        return model

    def evaluate(self, model, features, support, labels, mask, placeholders, epoch):  #model evaluate
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = self.sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)

        if epoch == 400:
            tsne = TSNE(n_components=2)
            x_tsne = tsne.fit_transform(outs_val[2])
            pre = model.predict()
            preVal = self.sess.run(pre, feed_dict=feed_dict_val)
            preLabel = np.argmax(preVal, axis=1)

            dataframe = pd.DataFrame(
                {'x0': x_tsne[:, 0], 'x1': x_tsne[:, 1],
                 'c': preLabel})  # save data
            dataframe.to_csv('scatter0AvgResult.csv', index=False, sep=',')
            fig = plt.figure()
            plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=preLabel, label="t-SNE")
            fig.savefig('scatter0Avg.png')
            plt.show()

        return outs_val[0], outs_val[1], (time.time() - t_test)

    def scale(self, intermediate_layer_output):

        for input_id in range(intermediate_layer_output.shape[0]):
            currentIdOutput = intermediate_layer_output[input_id]
            min = intermediate_layer_output[input_id].min()
            max = intermediate_layer_output[input_id].max()

            if min == 0 and max == 0:
                intermediate_layer_output[input_id] = 0
            else:
                intermediate_layer_output[input_id] = (currentIdOutput - min) / (max - min)


        return intermediate_layer_output

    def getSampleConverage(self, scaleOutVal, threshold):
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

    def getNeuronConverageFrequency(self, samplesCoverage):
        num_input, num_neuron = samplesCoverage.shape
        neuronCoverage = np.sum(samplesCoverage, axis=0) / num_input

        return neuronCoverage

    def getWeightCoverageFrequency(self, inputCoverage, outputCoverage):
        feature_num = len(inputCoverage)
        neuron_num = len(outputCoverage)

        weightCoverageFrequency = np.zeros(shape=(feature_num, neuron_num), dtype=np.float)

        for i in range(feature_num):
            for j in range(neuron_num):
                weightCoverageFrequency[i][j] = inputCoverage[i] + outputCoverage[j]

        return weightCoverageFrequency


    def getImportWeight(self, weightCoverage, weight, threshold):    #need to check
        weightRow = weightCoverage.shape[0]
        weightCol = weightCoverage.shape[1]

        #get the index ascending by column
        bb = weightCoverage[:, 0]
        min0 = np.min(bb)
        index0 = list(bb).index(min0)

        cc = weightCoverage[:, 1]
        min1 = np.min(cc)
        index1 = list(cc).index(min1)

        weightCoverageAscendingScore = np.argsort(weightCoverage, axis=0)
        weightAscendingScore = np.argsort(weight, axis=0)

        ord = weightCoverageAscendingScore - weightAscendingScore

        result = np.zeros(shape=(weightRow, weightCol), dtype=np.uint8)

        for i in range(weightRow):
            for j in range(weightCol):
                if ord[i][j] > threshold:
                    result[i][j] = 1

        return result

    def getDenseMatrix(self, sparseMatrix):
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

    def getImportPos(self, model, support, feed_dict):
        weight0 = self.sess.run(model.layers[0].vars['weights_0'])  # get the weight from tf.variable
        weight1 = self.sess.run(model.layers[1].vars['weights_0'])

        input = model.activations[0]  #the input of the model
        out0 = model.activations[1]  # the output of first layer
        out1 = model.activations[2]  # the output of second layer

        inputValSparseMatrix = self.sess.run(input, feed_dict=feed_dict)                     # get the input of the model
        currentInputVal = self.getDenseMatrix(inputValSparseMatrix)
        currentSupport = self.getDenseMatrix(support[0])
        out0Val = self.sess.run(out0, feed_dict=feed_dict)  # get the output of first layer
        out1Val = self.sess.run(out1, feed_dict=feed_dict)  # get the output of second layer

        inputVal = np.dot(currentSupport,  currentInputVal)
        scaleInputVal = self.scale(inputVal)          #scale into 0~1
        scaleOut0Val = self.scale(out0Val)            #scale into 0~1
        scaleOut1Val = self.scale(out1Val)            #scale into 0~1

        #---------get sample converage----------#
        print("get samples converage.............")
        inputCoverage = self.getSampleConverage(scaleInputVal, 0.0)
        samplesCoverageLayer0 = self.getSampleConverage(scaleOut0Val, 0.2)
        samplesCoverageLayer1 = self.getSampleConverage(scaleOut1Val, 0.2)

        #--------get neuraon converage-----------#
        print("get neuron converage frequency----------")
        neuronInputCoverageFrequency = self.getNeuronConverageFrequency(inputCoverage)
        neuronLayer0CoverageFrequency = self.getNeuronConverageFrequency(samplesCoverageLayer0)
        neuronLayer1CoverageFrequency = self.getNeuronConverageFrequency(samplesCoverageLayer1)

        #-------get weight converage------------#
        print("get weight converage---------------")
        weightLayer0CoverageFrequency = self.getWeightCoverageFrequency(neuronInputCoverageFrequency, neuronLayer0CoverageFrequency)
        weightLayer1CoverageFrequency = self.getWeightCoverageFrequency(neuronLayer0CoverageFrequency, neuronLayer1CoverageFrequency)


        #------get import weight---------------#   this place need to check
        print("get import weight pos-------------")
        weightLayer0ImportPos = self.getImportWeight(weightLayer0CoverageFrequency, weight0, -100)
        weightLayer1ImportPos = self.getImportWeight(weightLayer1CoverageFrequency, weight1, -5)

        print('the process is end')

        return weightLayer0ImportPos, weightLayer1ImportPos

    def modeTrain(self, model, FLAGS, features, y_val, val_mask,
                  support, y_train, train_mask, placeholders, epoch):
        t = time.time()

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = self.sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        a = self.sess.run(model.layers[0].vars['weights_0'])  # get the weight from tf.variable
        b = self.sess.run(model.layers[1].vars['weights_0'])

        # weightLayer0ImportPos, weightLayer1ImportPos = self.getImportPos(model, support, feed_dict)
        weightLayer0ImportPos = 0
        weightLayer1ImportPos = 0

        # Validation
        cost, acc, duration = self.evaluate(model, features, support, y_val, val_mask, placeholders, epoch)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        self.trainLoss.append(outs[1])
        self.trainAcc.append(outs[2])
        self.valLoss.append(cost)
        self.valAcc.append(acc)

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")

        return weightLayer0ImportPos, weightLayer1ImportPos

    def modeTest(self, model, features, support, y_test, test_mask, placeholders, epoch):
        # Testing
        test_cost, test_acc, test_duration = self.evaluate(model, features, support, y_test, test_mask, placeholders, epoch)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


class SecCoordinator:
    def __init__(self, host, port, n):
        self.n = n
        self.host = host
        self.port = port
        self.ready_client_ids = set()
        self.app = Flask(__name__)
        self.socketCoordinatorio = flask_socketio.SocketIO(self.app)
        self.register_Coordinator_handles()
        self.gcn = GcnMode()
        self.essList = []
        self.fssList = []
        self.trainEpoch = 400
        self.epoch = 0
        self.b = 0.005
        self.numkeys = 0
        self.rcvNum = 0
        self.xss0 = 0
        self.yss0 = 0
        self.ass0 = 0
        self.bss0 = 0
        self.css0 = 0
        self.ess0 = 0
        self.fss0 = 0
        self.zss0 = 0
        self.assM0 = 0
        self.bssM0 = 0
        self.cssM0 = 0
        self.e0 = 0
        self.f0 = 0
        self.zss0Sum = 0
        self.tempE0 = 0
        self.tempF0 = 0
        self.abc0Count = 0
        self.ef0Count = 0
        self.xss1 = 0
        self.yss1 = 0
        self.ass1 = 0
        self.bss1 = 0
        self.css1 = 0
        self.ess1 = 0
        self.fss1 = 0
        self.zss1 = 0
        self.assM1 = 0
        self.bssM1 = 0
        self.cssM1 = 0
        self.e1 = 0
        self.f1 = 0
        self.zss1Sum = 0
        self.abc1Count = 0
        self.ef1Count = 0
        self.tempE1 = 0
        self.tempF1 = 0
        self.weight0 = 0
        self.weight1 = 0
        self.weight00 = 0
        self.weight11 = 0

        self.pos = 0
        self.zssSumCount = 0
        self.tempZss0 = 0
        self.tempZss1 = 0
        self.zssCount = 0
        self.flag = True
        self.iterCount = 0

        self.clip0 = 4
        self.eps0 = 4
        self.delta0 = 0.00001
        self.sensitivity0 = 4
        self.clip1 = 4
        self.eps1 = 4
        self.delta1 = 0.00001
        self.sensitivity1 = 4
        self.sendAbc0StartTime = 0
        self.sendAbc0EndTime = 0
        self.multiplyss0StartTime = 0
        self.multiplyss0EndTime = 0
        self.ef0SendEndTime = 0
        self.ef0SendStartTime = 0


    def convertToArray0(self, dataCoo):
        data = dataCoo[0][1]
        row = dataCoo[0][0][:, -1]
        col = dataCoo[0][0][:, 0]

        currentCss = sp.coo_matrix((data, (row, col)), shape=(dataCoo[0][2][0], dataCoo[0][2][1]))
        cssArray = currentCss.toarray()

        return cssArray

    def convertToArray1(self, dataCoo):
        data = dataCoo[0][1]
        col = dataCoo[0][0][:, -1]
        row = dataCoo[0][0][:, 0]

        currentCss = sp.coo_matrix((data, (row, col)), shape=(dataCoo[0][2][0], dataCoo[0][2][1]))
        cssArray = currentCss.toarray()

        return cssArray

    def initGcn(self):   #init parameters and build model
        self.FLAGS = self.gcn.init_default_parameter()  #init parameters
        self.adj, self.features, self.allFeatures, self.y_train, self.y_val, self.y_test, \
        self.train_mask, self.val_mask, self.test_mask, \
        self.samplingGlobalAdjWithReduceNode, self.sampling_idx_range = self.gcn.load_cora(self.FLAGS) #load data
        # self.samplingGlobalAdjWithReduceNode = self.getNormalizedAdj()
        self.num_supports, self.supports, self.features, self.model_func = self.gcn.pre_process(self.FLAGS, self.features, self.adj) #pre_process
        # self.xss0 = self.convertToArray0(self.supports)
        self.xss0 = self.samplingGlobalAdjWithReduceNode
        # samplingIndex = random.sample(range(0, totalSampleNum), samplingNum)  # get random samplingIndex
        # dataframe = pd.DataFrame(self.samplingGlobalAdjWithReduceNode)  # save the samplingIndex of every client
        # dataframe.to_csv('0.csv', index=False, sep=',')  #

        self.yss0 = np.random.rand(self.xss0.shape[0], self.xss0.shape[1])
        # self.yss0 = np.dot(self.yss0, np.transpose(self.yss0))
        self.xss1 = self.xss0
        # self.supports = [preprocess_adj(self.adj)]
        self.model = self.gcn.modeBuild(self.num_supports, self.features, self.y_train, self.model_func, self.FLAGS)  # construct model
        self.gcn.sess.run(tf.global_variables_initializer())

    def getImportPos(self, model, support, feed_dict):
        weight0 = self.sess.run(model.layers[0].vars['weights_0'])  # get the weight from tf.variable
        weight1 = self.sess.run(model.layers[1].vars['weights_0'])

        input = model.activations[0]  # the input of the model
        out0 = model.activations[1]  # the output of first layer
        out1 = model.activations[2]  # the output of second layer

        inputValSparseMatrix = self.sess.run(input, feed_dict=feed_dict)  # get the input of the model
        currentInputVal = getDenseMatrix(inputValSparseMatrix)
        currentSupport = getDenseMatrix(support[0])
        out0Val = self.sess.run(out0, feed_dict=feed_dict)  # get the output of first layer
        out1Val = self.sess.run(out1, feed_dict=feed_dict)  # get the output of second layer

        inputVal = np.dot(currentSupport, currentInputVal)
        scaleInputVal = scale(inputVal)  # scale into 0~1
        scaleOut0Val = scale(out0Val)  # scale into 0~1
        scaleOut1Val = scale(out1Val)  # scale into 0~1

        # ---------get sample converage----------#
        print("get samples converage.............")
        inputCoverage = getSampleConverage(scaleInputVal, 0.0)
        samplesCoverageLayer0 = getSampleConverage(scaleOut0Val, 0.2)
        samplesCoverageLayer1 = getSampleConverage(scaleOut1Val, 0.2)

        # --------get neuraon converage-----------#
        print("get neuron converage frequency----------")
        neuronInputCoverageFrequency = getNeuronConverageFrequency(inputCoverage)
        neuronLayer0CoverageFrequency = getNeuronConverageFrequency(samplesCoverageLayer0)
        neuronLayer1CoverageFrequency = getNeuronConverageFrequency(samplesCoverageLayer1)

        # -------get weight converage------------#
        print("get weight converage---------------")
        weightLayer0CoverageFrequency, all_weight_coverage0 = getWeightCoverageFrequency(neuronInputCoverageFrequency,
                                                                                         neuronLayer0CoverageFrequency)
        weightLayer1CoverageFrequency, all_weight_coverage1 = getWeightCoverageFrequency(neuronLayer0CoverageFrequency,
                                                                                         neuronLayer1CoverageFrequency)

        # ------get import weight---------------#   this place need to check
        print("get import weight pos-------------")
        weightLayer0ImportPos = getImportWeight(all_weight_coverage0, weight0, 0.5, 0)
        weightLayer1ImportPos = getImportWeight(all_weight_coverage1, weight1, 0.5, 0)

        print('the process is end')

        return weightLayer0ImportPos, weightLayer1ImportPos

    def trainModel(self):
        print("train local model......")
        if self.epoch != self.trainEpoch:   #train the model
            weightLayer0ImportPos, weightLayer1ImportPos = self.gcn.modeTrain(self.model, self.FLAGS, self.features, self.y_val, self.val_mask,
                      self.supports, self.y_train, self.train_mask, self.model.placeholders, self.epoch)
            self.epoch += 1

            self.weight0 = self.gcn.sess.run(self.model.layers[0].vars['weights_0'])   # the weight of layer0
            self.weight1 = self.gcn.sess.run(self.model.layers[1].vars['weights_0'])   # the weight of layer1

            # noisedWeight0 = getAllNoiseGradient(self.weight0, self.eps0, self.delta0, self.sensitivity0)
            # noisedWeight1 = getAllNoiseGradient(self.weight1, self.eps1, self.delta1, self.sensitivity1)

            # noisedWeight0 = getNoiseForPartialGradient(self.weight0, weightLayer0ImportPos, self.eps0, self.delta0, self.sensitivity0)
            # noisedWeight1 = getNoiseForPartialGradient(self.weight1, weightLayer1ImportPos, self.eps1, self.delta1, self.sensitivity1)

            # self.weight0 = noisedWeight0
            # self.weight1 = noisedWeight1

            if self.epoch == self.trainEpoch:
                print("the training is ended, and then test the model.......")

                dataframe = pd.DataFrame(
                    {'train_loss': self.gcn.trainLoss, 'train_acc': self.gcn.trainAcc,
                     'val_loss': self.gcn.valLoss, 'val_acc': self.gcn.valAcc})  # save data
                dataframe.to_csv('client0AvgTrainResult.csv', index=False, sep=',')

                self.gcn.modeTest(self.model, self.features, self.supports, self.y_test, self.test_mask,
                                  self.model.placeholders, self.epoch)

        else:      #test the model
            dataframe = pd.DataFrame(
                {'train_loss': self.gcn.trainLoss, 'train_acc': self.gcn.trainAcc,
                 'val_loss': self.gcn.valLoss, 'val_acc': self.gcn.valAcc})  # save data
            dataframe.to_csv('client0TrainResult.csv', index=False, sep=',')

            print("the training is ended, and then test the model.......")
            self.gcn.modeTest(self.model, self.features, self.supports, self.y_test, self.test_mask, self.model.placeholders, self.epoch)


    def getAdjSupport(self, aggeratedAdj):
        currentAdj = collections.defaultdict(list)
        aggeratedAdj = np.round(aggeratedAdj, 3).tolist()

        count = 0
        for idx in self.sampling_idx_range:
            eachRow = aggeratedAdj[idx]
            eachList = []
            rowIndexs = [i for i, x in enumerate(eachRow) if x != 0]  # the index which not equal zero

            for rowIdx in rowIndexs:
                withInFlag = rowIdx in self.sampling_idx_range
                if withInFlag:
                    currentPos = self.sampling_idx_range.index(rowIdx)
                    eachList.append(currentPos)
            currentAdj[count] = eachList
            count += 1

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(currentAdj))
        support = [preprocess_adj(adj)]

        return support

    def SSSplit(self, data, clientNum):
        r = np.array([np.random.uniform(0, 4, (data.shape[0], data.shape[1])) for i in range(clientNum - 1)])
        data = data.astype('float64')
        data -= np.sum(r, axis=0).astype('float64')
        data = np.expand_dims(data, axis=0)
        dataList = np.concatenate([r, data], axis=0)
        return dataList

    def getSS(self, row0, col0, row1, col1, clientNum):
        a = np.random.rand(row0, col0)
        b = np.random.rand(row1, col1)
        c = a * b
        ass = self.SSSplit(a, clientNum)  # 秘密分享
        bss = self.SSSplit(b, clientNum)
        css = self.SSSplit(c, clientNum)

        return ass, bss, css

    def multiplyss0(self, row0, col0, row1, col1, clientNum):
        ass, bss, css = self.getSS(row0, col0, row1, col1, clientNum)  # get secret share

        ess = self.xss0 - ass[self.n]
        fss = self.yss0 - bss[self.n]

        return ess, fss, ass, bss, css

    def multiplyss1(self, row0, col0, row1, col1, clientNum):
        ass, bss, css = self.getSS(row0, col0, row1, col1, clientNum)  # get secret share

        ess = self.xss1 - ass[self.n]
        fss = self.yss1 - bss[self.n]

        return ess, fss, ass, bss, css

    def train(self):
        # self.supports = self.getAdjSupport(self.zss0Sum)
        self.trainModel()

        self.weight00 += self.weight0
        self.weight11 += self.weight1

        self.weight00 = self.weight00 / (self.n + 1)
        self.weight11 = self.weight11 / (self.n + 1)

        self.weight0 = self.weight00
        self.weight1 = self.weight11

        print('send the agg to every clients and start the next training')

        aggeratedMsg = {
            'aggeratedWeight0': self.weight00.tolist(),
            'aggeratedWeight1': self.weight11.tolist(),
        }

        for clientId in self.ready_client_ids:
            self.socketCoordinatorio.emit('dispatchParameters', aggeratedMsg, room=clientId)

        self.rcvNum = 0
        self.weight00 = 0
        self.weight11 = 0

    def selfTrain(self):
        localWeight0 = self.gcn.sess.run(self.model.layers[0].vars['weights_0'])
        localWeight1 = self.gcn.sess.run(self.model.layers[1].vars['weights_0'])

        # currentWeight0 = localWeight0
        # currentWeight1 = localWeight1

        currentWeight0 = self.weight0
        currentWeight1 = self.weight1

        assignWeight0 = self.model.layers[0].vars['weights_0'].assign(currentWeight0)
        assignWeight1 = self.model.layers[1].vars['weights_0'].assign(currentWeight1)

        self.gcn.sess.run(assignWeight0)  # assign aggerated weights for client
        self.gcn.sess.run(assignWeight1)

        self.trainModel()

        self.weight00 += self.weight0
        self.weight11 += self.weight1

        self.weight00 = self.weight00 / (self.n + 1)
        self.weight11 = self.weight11 / (self.n + 1)

        self.weight0 = self.weight00
        self.weight1 = self.weight11

        print('send the agg to every clients and start the next training')

        aggeratedMsg = {
            'aggeratedWeight0': self.weight00.tolist(),
            'aggeratedWeight1': self.weight11.tolist(),
        }

        for clientId in self.ready_client_ids:
            self.socketCoordinatorio.emit('dispatchParameters', aggeratedMsg, room=clientId)

        self.rcvNum = 0
        self.weight00 = 0
        self.weight11 = 0


    def sendAbc0SS(self, ass, bss, css):
        eachSendNum = 100
        shape = ass[0].shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.abc0Count < iterNum:
            print('range:', startPos, '----', endPos)
            keyList = list(self.ready_client_ids)
            for clientId in self.ready_client_ids:
                idx = keyList.index(clientId)
                eachSsMsg = {
                    'ass': np.array(ass[idx][startPos:endPos]).tolist(),
                    'bss': np.array(bss[idx][startPos:endPos]).tolist(),
                    'css': np.array(css[idx][startPos:endPos]).tolist(),
                }

                self.socketCoordinatorio.emit("multiplyss0", eachSsMsg, room=clientId)
            self.pos += eachSendNum
            self.abc0Count += 1

    def sendAbc1SS(self, ass, bss, css):
        eachSendNum = 100
        shape = ass[0].shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.abc1Count < iterNum:
            print('range:', startPos, '----', endPos)
            keyList = list(self.ready_client_ids)
            for clientId in self.ready_client_ids:
                idx = keyList.index(clientId)
                eachSsMsg = {
                    'ass': np.array(ass[idx][startPos:endPos]).tolist(),
                    'bss': np.array(bss[idx][startPos:endPos]).tolist(),
                    'css': np.array(css[idx][startPos:endPos]).tolist(),
                }

                self.socketCoordinatorio.emit("multiplyss1", eachSsMsg, room=clientId)
            self.pos += eachSendNum
            self.abc1Count += 1


    def sendEf0(self, e0, f0):
        eachSendNum = 100
        shape = e0.shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.ef0Count < iterNum:
            print('range:', startPos, '----', endPos)
            for clientId in self.ready_client_ids:
                eachSsMsg = {
                    'e': np.array(e0[startPos:endPos]).tolist(),
                    'f': np.array(f0[startPos:endPos]).tolist(),
                }

                self.socketCoordinatorio.emit("ef0", eachSsMsg, room=clientId)
            self.pos += eachSendNum
            self.ef0Count += 1

    def sendEf1(self, e1, f1):
        eachSendNum = 100
        shape = e1.shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.ef1Count < iterNum:
            print('range:', startPos, '----', endPos)
            for clientId in self.ready_client_ids:
                eachSsMsg = {
                    'e': np.array(e1[startPos:endPos]).tolist(),
                    'f': np.array(f1[startPos:endPos]).tolist(),
                }

                self.socketCoordinatorio.emit("ef1", eachSsMsg, room=clientId)
            self.pos += eachSendNum
            self.ef1Count += 1


    def sendZssSum(self, zss0Sum):
        eachSendNum = 100
        shape = zss0Sum.shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.zssSumCount < iterNum:
            print('range:', startPos, '----', endPos)
            for clientId in self.ready_client_ids:
                eachSsMsg = {
                    'zss0Sum': np.array(zss0Sum[startPos:endPos]).tolist(),
                }
                self.socketCoordinatorio.emit("zssSum", eachSsMsg, room=clientId)

            self.pos += eachSendNum
            self.zssSumCount += 1

    def boardcastHeartbeat(self):
        if self.numkeys == self.n:
            while(self.flag):
                for sid in self.ready_client_ids:
                    print('send heartbeat to', sid)
                    self.socketCoordinatorio.emit('heartbeat', 0, room=sid)
                    time.sleep(5)

    def register_Coordinator_handles(self):
        @self.socketCoordinatorio.on("wakeup")
        def handle_wakeup():
            print("Recieved wakeup from", request.sid)
            self.numkeys += 1

            if self.numkeys == self.n:
                print('All clients connected, send secret to every clients')

                #-----------FedAvg and LocalM-------------------#
                for clientId in self.ready_client_ids:
                    self.socketCoordinatorio.emit("startTrain", room=clientId)

                #------FedMG-------------------------------------#
                # self.thread = threading.Thread(target=self.boardcastHeartbeat)
                # self.thread.start()
                # #
                # self.multiplyss0StartTime = time.clock()
                # ess, fss, ass, bss, css = self.multiplyss0(self.xss0.shape[0], self.xss0.shape[1], self.yss0.shape[0], self.yss0.shape[1], self.n + 1) # get secret share
                # self.multiplyss0EndTime = time.clock()
                #
                # print('multiplyss0Time:', self.multiplyss0EndTime-self.multiplyss0StartTime)
                #
                # self.ess0 = ess
                # self.fss0 = fss
                # self.ass0 = ass[self.n]
                # self.bss0 = bss[self.n]
                # self.css0 = css[self.n]
                # self.assM0 = ass
                # self.bssM0 = bss
                # self.cssM0 = css
                #
                # self.sendAbc0StartTime = time.clock()
                # self.sendAbc0SS(self.assM0, self.bssM0, self.cssM0)

        @self.socketCoordinatorio.on("continueAbc0SS")
        def handle_continue_send_abc0ss():
            print('continueAbc0SS----', request.sid)
            self.rcvNum += 1

            if self.rcvNum == self.n:
               self.sendAbc0SS(self.assM0, self.bssM0, self.cssM0)
               self.rcvNum = 0

        @self.socketCoordinatorio.on("continueAbc1SS")
        def handle_continue_send_abc1ss():
            print('continueAbc1SS----', request.sid)
            self.rcvNum += 1

            if self.rcvNum == self.n:
                self.sendAbc1SS(self.assM1, self.bssM1, self.cssM1)
                self.rcvNum = 0

        @self.socketCoordinatorio.on("continueEf0")
        def handle_continue_send_ef0():
            print('continueEf0----', request.sid)
            self.rcvNum += 1

            if self.rcvNum == self.n:
                self.sendEf0(self.e0, self.f0)
                self.rcvNum = 0

        @self.socketCoordinatorio.on("continueEf1")
        def handle_continue_send_ef1():
            print('continueEf1----', request.sid)
            self.rcvNum += 1

            if self.rcvNum == self.n:
                self.sendEf1(self.e1, self.f1)
                self.rcvNum = 0

        @self.socketCoordinatorio.on("continueZssSum")
        def handle_continue_send_zss():
            print('continueZssSum----', request.sid)
            self.rcvNum += 1

            if self.rcvNum == self.n:
                self.sendZssSum(self.zss0Sum)
                self.rcvNum = 0


        @self.socketCoordinatorio.on("connect")
        def handle_connect():
            print(request.sid, " Connected")
            self.ready_client_ids.add(request.sid)
            print('Connected devices:', self.ready_client_ids)

        @self.socketCoordinatorio.on('disconnect')
        def handle_disconnect():
            print(request.sid, " Disconnected")
            if request.sid in self.ready_client_ids:
                self.ready_client_ids.remove(request.sid)
            print(self.ready_client_ids)

        @self.socketCoordinatorio.on("ssmultiply")
        def handle_ssmultiply():
            print('start another multiply ss')
            self.rcvNum += 1

            if self.rcvNum == self.n:
                ess, fss, ass, bss, css = self.multiplyss1(self.xss1.shape[0], self.xss1.shape[1], self.yss1.shape[0], self.yss1.shape[1], self.n + 1)  # get secret share

                self.ess1 = ess
                self.fss1 = fss
                self.ass1 = ass[self.n]
                self.bss1 = bss[self.n]
                self.css1 = css[self.n]
                self.assM1 = ass
                self.bssM1 = bss
                self.cssM1 = css

                self.rcvNum = 0
                self.pos = 0
                self.sendAbc1SS(self.assM1, self.bssM1, self.cssM1)

        @self.socketCoordinatorio.on("efss0")
        def handle_efss0(*args):
            print(self.rcvNum, "-------Recieved efss0 from", request.sid)
            self.sendAbc0EndTime = time.clock()
            print('Abc0 transmission time:', self.sendAbc0EndTime - self.sendAbc0StartTime)
            msg = args[0]

            eachSendNum = 100
            shape = self.ass0[0].shape
            iterNum = int(shape[0] / eachSendNum)

            if shape[0] % eachSendNum != 0:
                iterNum += 1

            currentEss = msg['ess']
            currentFss = msg['fss']

            self.tempE0 += np.array(currentEss)
            self.tempF0 += np.array(currentFss)

            self.rcvNum += 1

            if self.rcvNum == self.n:
                self.ef0Count += 1
                if self.ef0Count == 1:
                    self.e0 = np.array(self.tempE0).tolist()
                    self.f0 = np.array(self.tempF0).tolist()
                    self.socketCoordinatorio.emit("continueEf0SS")
                elif self.ef0Count < iterNum:
                    self.e0 += np.array(self.tempE0).tolist()
                    self.f0 += np.array(self.tempF0).tolist()
                    self.socketCoordinatorio.emit("continueEf0SS")
                else:
                    self.ef0Count = 0

                    self.efzStartTime = time.clock()

                    self.e0 += np.array(self.tempE0).tolist()  #add left e0
                    self.f0 += np.array(self.tempF0).tolist()

                    e0 = np.array(self.e0) + np.array(self.ess0)  #add coornator ess0
                    f0 = np.array(self.f0) + np.array(self.fss0)

                    self.e0 = e0
                    self.f0 = f0

                    self.zss0 = np.array(self.e0) * np.array(self.f0) + np.array(self.f0) * np.array(self.ass0) + np.array(self.e0) * np.array(self.bss0) + np.array(self.css0)
                    # self.zss0Sum = self.zss0
                    self.yss1 = self.zss0

                    self.efzEndTime = time.clock()
                    print('ef0 and zss0 computation time:', self.efzEndTime - self.efzStartTime)

                    self.pos = 0

                    self.ef0SendStartTime = time.clock()
                    self.sendEf0(self.e0, self.f0)

                self.tempE0 = 0
                self.tempF0 = 0
                self.rcvNum = 0

        @self.socketCoordinatorio.on("efss1")
        def handle_efss1(*args):
            print("Recieved efss1 from", request.sid)
            msg = args[0]

            eachSendNum = 100
            shape = self.ass1[0].shape
            iterNum = int(shape[0] / eachSendNum)

            if shape[0] % eachSendNum != 0:
                iterNum += 1

            currentEss = msg['ess']
            currentFss = msg['fss']

            self.tempE1 += np.array(currentEss)
            self.tempF1 += np.array(currentFss)

            self.rcvNum += 1

            if self.rcvNum == self.n:
                self.ef1Count += 1
                if self.ef1Count == 1:
                    self.e1 = np.array(self.tempE1).tolist()
                    self.f1 = np.array(self.tempF1).tolist()
                    self.socketCoordinatorio.emit("continueEf1SS")
                elif self.ef1Count < iterNum:
                    self.e1 += np.array(self.tempE1).tolist()
                    self.f1 += np.array(self.tempF1).tolist()
                    self.socketCoordinatorio.emit("continueEf1SS")
                else:
                    self.ef1Count = 0
                    self.e1 += np.array(self.tempE1).tolist()  # add left e0
                    self.f1 += np.array(self.tempF1).tolist()

                    e1 = np.array(self.e1) + np.array(self.ess1)  # add coornator ess0
                    f1 = np.array(self.f1) + np.array(self.fss1)

                    self.e1 = e1
                    self.f1 = f1

                    self.zss1 = np.array(self.e1) * np.array(self.f1) + np.array(self.f1) * np.array(self.ass1) + np.array(self.e1) * np.array(self.bss1) + np.array(self.css1)
                    self.zss1Sum = self.zss1

                    self.pos = 0
                    self.sendEf1(self.e1, self.f1)

                self.tempE1 = 0
                self.tempF1 = 0
                self.rcvNum = 0


        @self.socketCoordinatorio.on("zss")
        def handle_zss(*args):
            print("Recieved zss from", request.sid)
            self.ef0SendEndTime = time.clock()
            print('ef0 transmission time:', self.ef0SendEndTime - self.ef0SendStartTime)
            rcvmsg = args[0]

            eachSendNum = 100
            shape = self.ass0[0].shape
            iterNum = int(shape[0] / eachSendNum)

            if shape[0] % eachSendNum != 0:
                iterNum += 1

            currentZss0 = rcvmsg['zss0']
            # currentZss1 = rcvmsg['zss1']

            self.tempZss0 += np.array(currentZss0)
            # self.tempZss1 += np.array(currentZss1)

            self.rcvNum += 1

            if self.rcvNum == self.n:
                self.zssCount += 1
                if self.zssCount == 1:
                    self.zss0Sum = np.array(self.tempZss0).tolist()
                    # self.zss1Sum = np.array(self.tempZss1).tolist()
                    self.socketCoordinatorio.emit("continueZss")
                elif self.zssCount < iterNum:
                    self.zss0Sum += np.array(self.tempZss0).tolist()
                    # self.zss1Sum += np.array(self.tempZss1).tolist()
                    self.socketCoordinatorio.emit("continueZss")
                else:
                    self.zssCount = 0
                    self.zss0Sum += np.array(self.tempZss0).tolist()  # add left e0
                    zss0Sum = np.array(self.zss0Sum) + np.array(self.zss0)

                    self.zss0Sum = zss0Sum                            # the final adjacent matrix

                    # dataframe = pd.DataFrame(self.zss0Sum)  # save the samplingIndex of every client
                    # dataframe.to_csv('zss0Sum.csv', index=False, sep=',')  #

                    self.pos = 0
                    self.sendZssSum(self.zss0Sum)

                self.rcvNum = 0
                self.tempZss0 = 0
                self.tempZss1 = 0


        @self.socketCoordinatorio.on('aggerateParameters')
        def handle_grad(*args):
            msg = args[0]
            print('Get weight from client:', request.sid)

            currentClietnWeight0 = msg['weight0']
            currentClietnWeight1 = msg['weight1']

            self.weight00 += np.array(currentClietnWeight0)
            self.weight11 += np.array(currentClietnWeight1)

            self.rcvNum += 1
            if self.rcvNum == self.n:
                if self.iterCount == 0:
                   self.thread = threading.Thread(target=self.train)
                   self.thread.start()
                else:
                    self.thread = threading.Thread(target=self.selfTrain)
                    self.thread.start()
                self.iterCount += 1


    def start(self):
        self.initGcn()
        self.socketCoordinatorio.run(self.app, host=self.host, port=self.port)


cost_val = []
if __name__=="__main__":
    print('client0')
    coordinator = SecCoordinator("127.0.0.1", 2019, 5)
    print("listening on 127.0.0.1:2019")
    coordinator.start()
