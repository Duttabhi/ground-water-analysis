import math
import numpy as np
import pygal
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

def lanalysis(inpe, arr):
	np.random.seed(7)

	#prepare data
	dataset = arr.reshape(-1,1)
	dataset = dataset.astype('float32')

	#normalize and transform data
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)	
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	#prepare model
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=inpe, batch_size=1, verbose=2)

	#predict results
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	trainPredict = scaler.inverse_transform(trainPredict.reshape(-1,1))
	testPredict = scaler.inverse_transform(testPredict.reshape(-1,1))
	dataset = scaler.inverse_transform(dataset)
	testPredictPlot=[None]*len(dataset)
	testPredictPlot[len(trainPredict):len(dataset)] = testPredict

	#plot results
	graph=pygal.Line(width=900, height=500, explicit_size=True)
	graph.title='Recurrent Neural Networks based predictions'
	graph.add('original', dataset)
	graph.add('train', trainPredict)
	graph.add('test', testPredictPlot)
	graph.range=[0, 100]
	graph_data=graph.render_data_uri()
	return graph_data

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)