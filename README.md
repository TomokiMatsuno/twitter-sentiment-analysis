# twitter-sentiment-analysis
A neural network model for sentiment analysis on a twitter dataset.

# Model description
This model receives a tweet of variable length and classifies it as either positive or negative.
The structure is just a simple Bi-LSTM plus Multi-Layer Perceptron Classifier.

# Performance
I ran an experiment of positive-negative classification on the Twitter Japanese Reputation Analysis Dataset distributed in the link below.

http://bigdata.naist.jp/~ysuzuki/data/twitter/

For validation set, I randomly selected 20% of the data with positive or negative label and used the rest for training.
The ratio of positive and negative labels is almost 1:2.

As a result, this model achieved 82% accuracy which is much higher than that of my Bag of Words based model whose accuracy is 63%.
