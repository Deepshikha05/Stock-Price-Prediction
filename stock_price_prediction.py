import tweepy
import csv
import numpy as np
import os
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def get_data(filename):
	dates = [] # for displaying X axis
	X = [] # features
	Y = [] # labels

	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		
		i = 0

		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			# import pdb; pdb.set_trace()
			X.append([int(row[0].split('-')[0]), polarities[i]])
			Y.append(float(row[1]))

			i = i + 1
		
	return dates, X, Y

# Predict prices using Support Vector Machines
def predict_prices_svr(dates, X, Y, x):
	# converting to matrix of n X 2
	X = np.reshape(X, (len(X), 2))
	x = np.reshape(x, (len(x), 2))

	# defining the support vector regression models
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	
	# fitting the data points in the models
	svr_rbf.fit(X, Y)
	svr_lin.fit(X, Y)
	svr_poly.fit(X, Y)
	
	plt.scatter(dates, Y, color='black', label='Data') # plotting the initial datapoints 
	plt.plot(dates, svr_rbf.predict(X), color='red', label='RBF model') # plotting the line made by the RBF kernel
	plt.plot(dates,svr_lin.predict(X), color='green', label='Linear model') # plotting the line made by linear kernel
	plt.plot(dates,svr_poly.predict(X), color='blue', label='Polynomial model') # plotting the line made by polynomial kernel
	
	# specifying labels
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_prices_nn(dates, X, Y, x):
	# converting to matrix of n X 2
	X = np.reshape(X, (len(X), 2))
	x = np.reshape(x, (len(x), 2))
	
	model = Sequential()
	model.add(Dense(64, input_dim=2, activation='relu'))
	model.add(Dense(8))
	model.add(Dense(1))

	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	# Fit the model
	model.fit(X, Y, epochs=150, batch_size=10)
	
	plt.scatter(dates, Y, color='black', label='Data') # plotting the initial datapoints 
	plt.plot(dates, model.predict(X), color= 'red', label='Neural Network') # plotting the line made by the Neural Network
	
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Neural Network Regression')
	
	plt.legend()
	plt.show()
	
	# evaluate the model
	scores = model.evaluate(X, Y)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	return model.predict(x)

consumer_key= 'OFAKWwY2Y9sGNxwkFVGQvX2Dp'
consumer_secret= 'dnhb34YHJBF3AHCr1KivrSXabguAHPEgvf0zwH01xCCeee0svx'

access_token='2997965748-PFhrOMf1aULaTrEqG40rsKl3Nz5Yb2Ee1f4yJzg'
access_token_secret='6TfkmxTnlLlSSOKo6xt4wXNW9jvDwCrwWq1bS5hHBc8Jx'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

public_tweets = tweepy.Cursor(api.search, q="Facebook", count=252).items()
polarities = []
i = 0

# getting 30 most recent tweets
for tweet in public_tweets:
	analysis = TextBlob(tweet.text)
	polarities.append(1 - int(analysis.sentiment.polarity < 0))
	
	i += 1

	if i == 252:
		break
dates, X, Y = get_data('fb.csv') # contains data of only 1 month

predicted_price_svr = predict_prices_svr(dates, X, Y, [[29, 1]])
print("RBF SVR -", predicted_price_svr[0])
print("Linear SVR -", predicted_price_svr[1])
print("Polynomial SVR -", predicted_price_svr[2])

predicted_price_nn = predict_prices_nn(dates, X, Y, [[29, 1]])
print("Neural Network -", predicted_price_nn)