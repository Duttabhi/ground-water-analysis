import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from pyFTS.models import chen
from pyFTS.partitioners import Grid
from sklearn.preprocessing import MinMaxScaler
import pygal

def fanalysis(inpp, arr):
	np.random.seed(7)

	dataset = arr.reshape(-1,1)
	dataset = dataset.astype('float32')
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	#prepare data
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	#partition data for suzzy logic
	fs = Grid.GridPartitioner(data=train, npart=inpp)
	model = chen.ConventionalFTS(partitioner=fs)
	model.fit(train)

	#train model
	trainPredict = model.predict(train, dtype=object)
	testPredict = model.predict(test, dtype=object)

	#predict results
	trainPredict = np.array(trainPredict)
	testPredict = np.array(testPredict)	
	trainPredict = scaler.inverse_transform(trainPredict.reshape(-1,1))
	testPredict = scaler.inverse_transform(testPredict.reshape(-1,1))
	dataset = scaler.inverse_transform(dataset)
	testPredictPlot=[None]*len(dataset)
	testPredictPlot[len(trainPredict):len(dataset)] = testPredict
	
	#plot graph
	graph=pygal.Line(width=900, height=500, explicit_size=True)
	graph.title='Fuzzy Logic based predictions'
	graph.add('original', dataset)
	graph.add('train', trainPredict)
	graph.add('test', testPredictPlot)
	graph.range=[0, 100]
	graph_data=graph.render_data_uri()
	return graph_data
