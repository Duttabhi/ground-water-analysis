from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
from fuzzyanalysis import fanalysis
from lstmanalysis import lanalysis

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html");

@app.route('/result', methods = ['POST'])
def results():
	try:
		arr = pd.read_csv(request.files.get('inpdata'), usecols=[0]).values
		modeltype=request.form['modelselect']
		if (modeltype=='1'):
			inpe=request.form['inpepoch']
			if(len(arr)>100 and len(arr)<2000 and bool(inpe)==True):		
				graph_data=lanalysis(int(inpe), arr)
				return render_template('results.html', msg='You have selected the LSTM model', graph_data=graph_data)
			else:
				return render_template('results.html', msg='Please fill all the fields and make sure the file has atleast 100 records but less than 2000')	
		elif (modeltype=='2'):
			inpp=request.form['inppart']
			if(len(arr)>100 and len(arr)<2000 and bool(inpp)==True):
				graph_data=fanalysis(int(inpp), arr)
				return render_template('results.html', msg='You have selected the fuzzy based model', graph_data=graph_data)
			else:
				return render_template('results.html', msg='Please fill all the fields and make sure the file has atleast 100 records but less than 2000')	

		else:
			return render_template('results.html', msg='Some error occured while performing analysis...')
	except Exception:
		return render_template('results.html', msg='Either some internal error occured or you have left few fields...')

if __name__  == '__main__':
	app.run(debug=True)
