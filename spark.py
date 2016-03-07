from pyspark import SparkContext
import json

sc = SparkContext()





if __name__ == '__main__':
    sc.textFile('data/twitter_dump.txt')\
    .filter(lambda tw: len(tw)>1)\
    .filter(lambda tw: 'created_at' in tw)\
    .map(lambda tw: tw.strip())\
    .map(lambda tw: json.loads(tw)['text'])\
    .take(1000)
