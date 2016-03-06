#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

# from parse_json import emoji_list

# read api keys from json files
with open('/home/han/.api_key/twitter_key.json') as f:
    keys = json.load(f)
    consumer_key = keys['consumer_key']
    consumer_secret = keys['consumer_secret']
    access_token = keys['access_token']
    access_token_secret = keys['access_secret']




with open('../filter.txt') as f:
    filters = [ keyword.strip() for keyword in f ]

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status

def getting_firehose():
    SF_AND_NY_BOX = [-122.75,36.8,-121.75,37.8,-74,40,-73,41]
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    while True:
        try:
            stream = Stream(auth, l)

            stream.filter(track = filters, locations = SF_AND_NY_BOX)
        except:
            continue

if __name__ == '__main__':

    SF_AND_NY_BOX = [-122.75,36.8,-121.75,37.8,-74,40,-73,41]

    #This handles Twitter authetification and the connection to Twitter Streaming API
    getting_firehose()


    # execute with python ../twitter_scrape.py >> twitter_dump.txt
