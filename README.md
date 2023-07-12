# Twitter-Sentiment-Analysis
A Sentiment Analysis Tool that plots the data regarding "#Covid19" since January 2022, using nums.py and Pandas.py 
Collects tweets containing the hashtag "#COVID19" since January 1, 2022, using the Twitter API.
Preprocesses the tweets by removing hyperlinks, hashtags, mentions, and performing lemmatization using the TextBlob library.
Performs sentiment analysis on the preprocessed tweets using TextBlob, and labels them as positive, neutral, or negative.
Splits the data into training and testing sets.
Vectorizes the text data using CountVectorizer from scikit-learn.
Trains a Random Forest classifier on the vectorized training data.
Predicts sentiment labels for the test set.
Calculates the accuracy of the sentiment predictions.
Performs further analysis:
Counts the number of tweets in each sentiment category.
Prints the sentiment distribution.
Plots the sentiment distribution using a bar chart.
Calculates the word frequencies for the entire dataset.
Prints the top 10 most frequent words.
Plots the top 20 most frequent words and their frequencies using a bar chart.
