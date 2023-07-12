import tweepy
import pandas as pd
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Twitter API credentials
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Collect tweets containing the hashtag "#COVID19" since January 1, 2022
tweets = tweepy.Cursor(api.search, q="#COVID19", since="2022-01-01").items()

# Create a list to store the preprocessed tweets
preprocessed_tweets = []

# Preprocess the tweets
for tweet in tweets:
    # Remove hyperlinks, hashtags, mentions
    preprocessed_tweet = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", tweet.text)
    
    # Lemmatize the remaining tokens
    preprocessed_tweet = " ".join(TextBlob(preprocessed_tweet).words.lemmatize())
    
    preprocessed_tweets.append(preprocessed_tweet)

# Create a pandas DataFrame to store the preprocessed tweets
data = pd.DataFrame(data=preprocessed_tweets, columns=["Preprocessed Tweets"])

# Perform sentiment analysis using TextBlob
data["Sentiment"] = data["Preprocessed Tweets"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Label the tweets as positive, neutral, or negative based on sentiment scores
data["Sentiment_Label"] = data["Sentiment"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Preprocessed Tweets"], data["Sentiment_Label"], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Random Forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train_vectorized, y_train)

# Predict sentiment labels for the test set
y_pred = classifier.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Further analysis

# Count the number of tweets in each sentiment category
sentiment_counts = data["Sentiment_Label"].value_counts()

# Print the sentiment distribution
print("Sentiment Distribution:")
print(sentiment_counts)

# Visualization

# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind="bar", color=["green", "blue", "red"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Word frequency analysis

# Vectorize the entire dataset
data_vectorized = vectorizer.transform(data["Preprocessed Tweets"])

# Get the word frequencies
word_frequencies = data_vectorized.sum(axis=0)

# Convert to a dictionary with words as keys and frequencies as values
word_frequencies_dict = {word: frequency for word, frequency in zip(vectorizer.get_feature_names(), word_frequencies.tolist()[0])}

# Sort the words by frequency in descending order
sorted_word_frequencies = sorted(word_frequencies_dict.items(), key=lambda x: x[1], reverse=True)

# Print the top 10 most frequent words
print("Top 10 Most Frequent Words:")
for word, frequency in sorted_word_frequencies[:10]:
    print(word, ":", frequency)

# Visualization of word frequency

# Get the top 20 most frequent words and their frequencies
top_20_words = sorted_word_frequencies[:20]
words = [word[0] for word in top_20_words]
frequencies = [word[1] for word in top_20_words]

# Plot the word frequencies
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies, color="purple")
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
