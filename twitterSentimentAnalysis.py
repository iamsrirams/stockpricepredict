import tweepy
from textblob import TextBlob

# Step 1 - Authenticate
consumer_key = 'qSwcoYsUrXiNodGFqmuQIol0u'
consumer_secret = 'fxKBJI7InOZDTq3f77FRUtokn1bzAvESOueVVNyYGZ9udy4SOl'

access_token = '1386261829-8lOeZE3ozK4OTpEvf4vdHhD9JgJOXdGui6JuOnu'
access_token_secret = 'DDQ40ITSSlztmmplb4iJ9wBuDW5kcr47m1bPSaU1r4mpL'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Step 3 - Retrieve Tweets
public_tweets = api.search('Trump')

# CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
# and label each one as either 'positive' or 'negative', depending on the sentiment
# You can decide the sentiment polarity threshold yourself

for tweet in public_tweets:
    #print()
    print(tweet.text)

    # Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    print("")
