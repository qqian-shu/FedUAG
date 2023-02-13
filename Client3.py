from __future__ import division
from __future__ import print_function

from socketIO_client import SocketIO, LoggingNamespace
from random import randrange
import tensorflow as tf
from .utils1 import *
from .models import GCN, MLP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import random
import torch

class GcnMode:
    def __init__(self):
        self.sess = tf.Session()
        self.trainLoss = []
        self.trainAcc = []
        self.valLoss = []
        self.valAcc = []

    def load_cora(self, FLAGS):  # load cora dataset
        adj, features, allFeatures, y_train, y_val, y_test, train_mask, \
        val_mask, test_mask, samplingGlobalAdjWithReduceNode, sampling_idx_range = load_data1(FLAGS.dataset, 0.5, 20,
                                                                                              True, 'client3.csv')

        return adj, features, allFeatures, y_train, y_val, y_test, train_mask, \
               val_mask, test_mask, samplingGlobalAdjWithReduceNode, sampling_idx_range

    def init_default_parameter(self):  # set parameters
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

    def pre_process(self, FLAGS, features, adj):  # some preprocess
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

    def modeBuild(self, num_supports, features, y_train, model_func, FLAGS):  # build train model
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

    def evaluate(self, model, features, support, labels, mask, placeholders, epoch):  # model evaluate
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
            dataframe.to_csv('scatter3AvgResult.csv', index=False, sep=',')
            fig = plt.figure()
            plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=preLabel, label="t-SNE")
            fig.savefig('scatter3Avg.png')
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



class secaggclient:
    def __init__(self, serverhost, serverport, mixData, poxyData):
        self.sio = SocketIO(serverhost, serverport, LoggingNamespace)
        self.gcn = GcnMode()
        self.keys = {}
        self.clientsId = []
        self.trainEpoch = 400
        self.epoch = 0
        self.b = 0.005
        self.xss0 = 0
        self.yss0 = 0
        self.ass0 = 0
        self.bss0 = 0
        self.css0 = 0
        self.zss0 = 0
        self.ess0 = 0
        self.fss0 = 0
        self.e0 = 0
        self.f0 = 0
        self.ef0Count = 0
        self.count0 = 0
        self.xss1 = 0
        self.yss1 = 0
        self.ass1 = 0
        self.bss1 = 0
        self.css1 = 0
        self.zss1 = 0
        self.ess1 = 0
        self.fss1 = 0
        self.e1 = 0
        self.f1 = 0
        self.ef1Count = 0
        self.count1 = 0
        self.pos = 0
        self.count = 0

        self.clip0 = 4
        self.eps0 = 4
        self.delta0 = 0.00001
        self.sensitivity0 = 4
        self.clip1 = 4
        self.eps1 = 4
        self.delta1 = 0.00001
        self.sensitivity1 = 4

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

    def getIterNum(self):
        eachSendNum = 100
        shape = self.samplingGlobalAdjWithReduceNode.shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        return iterNum

    def start(self):  # init parameters and build model
        self.FLAGS = self.gcn.init_default_parameter()  # init parameters
        self.adj, self.features, self.allFeatures, self.y_train, self.y_val, self.y_test, \
        self.train_mask, self.val_mask, self.test_mask, \
        self.samplingGlobalAdjWithReduceNode, self.sampling_idx_range = self.gcn.load_cora(
            self.FLAGS)  # load data
        self.iterNum = self.getIterNum()
        self.num_supports, self.supports, self.features, self.model_func = self.gcn.pre_process(self.FLAGS, self.features, self.adj)  # pre_process
        # self.xss0 = self.convertToArray0(self.supports)
        # dataframe = pd.DataFrame(self.samplingGlobalAdjWithReduceNode)  # save the samplingIndex of every client
        # dataframe.to_csv('3.csv', index=False, sep=',')  #

        self.xss0 = self.samplingGlobalAdjWithReduceNode
        self.yss0 = np.random.rand(self.xss0.shape[0], self.xss0.shape[1])
        # self.yss0 = self.convertToArray1([self.features])
        # self.yss0 = np.dot(self.yss0, np.transpose(self.yss0))
        # self.supports = [preprocess_adj(self.adj)]

        self.model = self.gcn.modeBuild(self.num_supports, self.features, self.y_train, self.model_func,
                                                   self.FLAGS)  # construct model
        self.gcn.sess.run(tf.global_variables_initializer())
        self.register_handles()
        print("Starting")
        self.sio.emit("wakeup")
        self.sio.wait()

    def trainModel(self):
        print("train local model......")
        if self.epoch != self.trainEpoch:  # train the model
            weightLayer0ImportPos, weightLayer1ImportPos = self.gcn.modeTrain(self.model, self.FLAGS, self.features, self.y_val, self.val_mask,
                                          self.supports, self.y_train, self.train_mask, self.model.placeholders,
                                          self.epoch)
            self.epoch += 1

            weight0 = self.gcn.sess.run(self.model.layers[0].vars['weights_0'])  # the weight of layer0
            weight1 = self.gcn.sess.run(self.model.layers[1].vars['weights_0'])  # the weight of layer1

            # noisedWeight0 = getAllNoiseGradient(weight0, self.eps0, self.delta0, self.sensitivity0)
            # noisedWeight1 = getAllNoiseGradient(weight1, self.eps1, self.delta1, self.sensitivity1)

            # noisedWeight0 = getNoiseForPartialGradient(self.weight0, weightLayer0ImportPos, self.eps0, self.delta0, self.sensitivity0)
            # noisedWeight1 = getNoiseForPartialGradient(self.weight1, weightLayer1ImportPos, self.eps1, self.delta1, self.sensitivity1)

            # send weight&adj to server.......
            print('send parameters to server......')
            # msg = {
            #     'weight0': noisedWeight0.tolist(),
            #     'weight1': noisedWeight1.tolist(),
            # }
            msg = {
                'weight0': weight0.tolist(),
                'weight1': weight1.tolist(),
            }
            self.sio.emit("aggerateParameters", msg)
        else:  # test the model
            print("the training is ended, and then test the model.......")

            dataframe = pd.DataFrame(
                {'train_loss': self.gcn.trainLoss, 'train_acc': self.gcn.trainAcc,
                 'val_loss': self.gcn.valLoss, 'val_acc': self.gcn.valAcc})  # save data
            dataframe.to_csv('client3AvgTrainResult.csv', index=False, sep=',')

            self.gcn.modeTest(self.model, self.features, self.supports, self.y_test, self.test_mask,
                                         self.model.placeholders, self.epoch)

    def preProcessAdj(self, aggeratedAdj):
        size = np.array(aggeratedAdj).shape

        for i in range(size[0]):
            for j in range(size[1]):
                if np.abs(aggeratedAdj[i][j]) < 0.000001:
                    aggeratedAdj[i][j] = 0

        return aggeratedAdj

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

    def sendEf0SS(self, ess, fss):
        eachSendNum = 100
        shape = ess.shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.count < iterNum:
            print('range:', startPos, '----', endPos)
            eachSsMsg = {
                    'ess': np.array(ess[startPos:endPos]).tolist(),
                    'fss': np.array(fss[startPos:endPos]).tolist(),
                }

            self.sio.emit("efss0", eachSsMsg)
            self.pos += eachSendNum
            self.count += 1

    def sendEf1SS(self, ess, fss):
        eachSendNum = 100
        shape = ess.shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.count1 < iterNum:
            print('range:', startPos, '----', endPos)
            eachSsMsg = {
                    'ess': np.array(ess[startPos:endPos]).tolist(),
                    'fss': np.array(fss[startPos:endPos]).tolist(),
                }

            self.sio.emit("efss1", eachSsMsg)
            self.pos += eachSendNum
            self.count1 += 1

    def sendZss(self, zss0):
        eachSendNum = 100
        shape = zss0.shape
        iterNum = int(shape[0] / eachSendNum)

        if shape[0] % eachSendNum != 0:
            iterNum += 1

        startPos = self.pos
        endPos = self.pos + eachSendNum

        if shape[0] - startPos < eachSendNum:
            endPos = shape[0]

        if self.count < iterNum:
            print('range:', startPos, '----', endPos)
            eachSsMsg = {
                'zss0': np.array(zss0[startPos:endPos]).tolist(),
            }

            self.sio.emit("zss", eachSsMsg)
            self.pos += eachSendNum
            self.count += 1

    # def sendZss(self, zss0, zss1):
    #     eachSendNum = 100
    #     shape = zss0.shape
    #     iterNum = int(shape[0] / eachSendNum)
    #
    #     if shape[0] % eachSendNum != 0:
    #         iterNum += 1
    #
    #     startPos = self.pos
    #     endPos = self.pos + eachSendNum
    #
    #     if shape[0] - startPos < eachSendNum:
    #         endPos = shape[0]
    #
    #     if self.count < iterNum:
    #         print('range:', startPos, '----', endPos)
    #         eachSsMsg = {
    #             'zss0': np.array(zss0[startPos:endPos]).tolist(),
    #             'zss1': np.array(zss1[startPos:endPos]).tolist(),
    #         }
    #
    #         self.sio.emit("zss", eachSsMsg)
    #         self.pos += eachSendNum
    #         self.count += 1

    def register_handles(self):
        def on_connect(*args):
            msg = args[0]
            self.sio.emit("connect")
            print("Connected and recieved this message", msg['message'])

        def on_disconnect(*args):
            print("Disconnected")
            self.sio.emit("disconnect")

        def on_get_ef0(*args):
            print('get ef0......')
            msg = args[0]

            currentE0 = msg['e']      # get e, f
            currentF0 = msg['f']

            # print('E0:', currentE0)
            #
            # print('F0', currentF0)

            self.ef0Count += 1

            if self.ef0Count == 1:
                self.e0 = np.array(currentE0).tolist()
                self.f0 = np.array(currentF0).tolist()
                self.sio.emit("continueEf0")
            elif self.ef0Count < self.iterNum:
                self.e0 += np.array(currentE0).tolist()
                self.f0 += np.array(currentF0).tolist()
                self.sio.emit("continueEf0")
            else:
                self.e0 += np.array(currentE0).tolist()
                self.f0 += np.array(currentF0).tolist()

                self.zss0 = np.array(self.f0) * np.array(self.ass0) + np.array(self.e0) * np.array(self.bss0) + np.array(self.css0)

                self.count = 0
                self.pos = 0
                self.sendZss(self.zss0)

                # self.xss1 = self.xss0
                # self.yss1 = self.zss0
                #
                # self.sio.emit("ssmultiply")


        def on_get_ef1(*args):
            print('get ef1......')
            msg = args[0]

            currentE1 = msg['e']  # get e, f
            currentF1 = msg['f']

            self.ef1Count += 1

            if self.ef1Count == 1:
                self.e1 = np.array(currentE1).tolist()
                self.f1 = np.array(currentF1).tolist()
                self.sio.emit("continueEf1")
            elif self.ef1Count < self.iterNum:
                self.e1 += np.array(currentE1).tolist()
                self.f1 += np.array(currentF1).tolist()
                self.sio.emit("continueEf1")
            else:
                self.e1 += np.array(currentE1).tolist()
                self.f1 += np.array(currentF1).tolist()

                self.zss1 = np.array(self.f1) * np.array(self.ass1) + np.array(self.e1) * np.array(self.bss1) + np.array(self.css1)

                self.count = 0
                self.pos = 0
                self.sendZss(self.zss0, self.zss1)


        def on_get_ss0(*args):
            print('get multiply secret share0......')
            msg = args[0]

            self.count0 += 1

            currentAss0 = msg['ass']
            currentBss0 = msg['bss']
            currentCss0 = msg['css']

            if self.count0 == 1:
                self.ass0 = np.array(currentAss0).tolist()
                self.bss0 = np.array(currentBss0).tolist()
                self.css0 = np.array(currentCss0).tolist()
                self.sio.emit("continueAbc0SS")
            elif self.count0 < self.iterNum:
                self.ass0 += np.array(currentAss0).tolist()
                self.bss0 += np.array(currentBss0).tolist()
                self.css0 += np.array(currentCss0).tolist()
                self.sio.emit("continueAbc0SS")
            else:
                self.ass0 += np.array(currentAss0).tolist()
                self.bss0 += np.array(currentBss0).tolist()
                self.css0 += np.array(currentCss0).tolist()

                self.ess0 = self.xss0 - np.array(self.ass0)
                self.fss0 = self.yss0 - np.array(self.bss0)

                self.pos = 0
                self.count0 = 0

                print('send ess0 and fss0 to coordinator.')

                self.sendEf0SS(self.ess0, self.fss0)


        def on_get_ss1(*args):
            print('get multiply secret share1......')
            msg = args[0]

            self.count1 += 1

            currentAss1 = msg['ass']
            currentBss1 = msg['bss']
            currentCss1 = msg['css']

            if self.count1 == 1:
                self.ass1 = np.array(currentAss1).tolist()
                self.bss1 = np.array(currentBss1).tolist()
                self.css1 = np.array(currentCss1).tolist()
                self.sio.emit("continueAbc1SS")
            elif self.count1 < self.iterNum:
                self.ass1 += np.array(currentAss1).tolist()
                self.bss1 += np.array(currentBss1).tolist()
                self.css1 += np.array(currentCss1).tolist()
                self.sio.emit("continueAbc1SS")
            else:
                self.ass1 += np.array(currentAss1).tolist()
                self.bss1 += np.array(currentBss1).tolist()
                self.css1 += np.array(currentCss1).tolist()

                self.ess1 = self.xss1 - np.array(self.ass1)
                self.fss1 = self.yss1 - np.array(self.bss1)

                self.pos = 0
                self.count1 = 0

                print('send ess1 and fss1 to coordinator.')

                self.sendEf1SS(self.ess1, self.fss1)

        def on_get_zssSum(*args):
            print('get zssSum secret share......')
            msg = args[0]

            self.count1 += 1

            currentZss0Sum = msg['zss0Sum']

            if self.count1 == 1:
                self.zss0Sum = np.array(currentZss0Sum).tolist()
                self.sio.emit("continueZssSum")
            elif self.count1 < self.iterNum:
                self.zss0Sum += np.array(currentZss0Sum).tolist()
                self.sio.emit("continueZssSum")
            else:
                self.zss0Sum += np.array(currentZss0Sum).tolist()
                self.pos = 0
                self.count1 = 0

                print('start training.....')
                self.supports = self.getAdjSupport(self.zss0Sum)
                self.trainModel()


        def on_agg_parameters(*args):
            print('get the aggerated parameters from server and train model.......')
            msg = args[0]

            globalWeight0 = msg['aggeratedWeight0']  # the global weight
            globalWeight1 = msg['aggeratedWeight1']

            # -------------------modify---------------------------------#
            localWeight0 = self.gcn.sess.run(self.model.layers[0].vars['weights_0'])
            localWeight1 = self.gcn.sess.run(self.model.layers[1].vars['weights_0'])

            # currentWeight0 = localWeight0
            # currentWeight1 = localWeight1

            currentWeight0 = globalWeight0
            currentWeight1 = globalWeight1

            assignWeight0 = self.model.layers[0].vars['weights_0'].assign(currentWeight0)
            assignWeight1 = self.model.layers[1].vars['weights_0'].assign(currentWeight1)

            self.gcn.sess.run(assignWeight0)  # assign aggerated weights for client
            self.gcn.sess.run(assignWeight1)

            self.trainModel()

        def on_heartbeat(*args):
            msg = args[0]
            print('recv heartbeat')

        def on_continueEf0SS(*args):
            print('continueEf0SS----')

            self.sendEf0SS(self.ess0, self.fss0)

        def on_continueEf1SS(*args):
            print('continueEf1SS----')

            self.sendEf1SS(self.ess1, self.fss1)

        def on_continueZss(*args):
            print('continueZss----')
            self.sendZss(self.zss0)
            # self.sendZss(self.zss0, self.zss1)

        def on_heartbeat(*args):
            msg = args[0]
            print('recv heartbeat')

        def on_train(*args):
            print('start training.....')
            self.trainModel()

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('dispatchParameters', on_agg_parameters)
        self.sio.on('heartbeat', on_heartbeat)
        self.sio.on('multiplyss0', on_get_ss0)
        self.sio.on('ef0', on_get_ef0)
        self.sio.on('multiplyss1', on_get_ss1)
        self.sio.on('ef1', on_get_ef1)
        self.sio.on('continueEf0SS', on_continueEf0SS)
        self.sio.on('continueEf1SS', on_continueEf1SS)
        self.sio.on('continueZss', on_continueZss)
        self.sio.on('zssSum', on_get_zssSum)
        self.sio.on('heartbeat', on_heartbeat)
        self.sio.on('startTrain', on_train)



cost_val = []
if __name__ == "__main__":
    print('client3')
    s = secaggclient("127.0.0.1", 2019, -3, 3)  # mixData:-3, -7, 10, poxyData:-3, 3
    s.start()
    print("Ready")