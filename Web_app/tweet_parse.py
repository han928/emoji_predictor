from __future__ import division
from pyspark import SparkContext
import json
import re
import nltk
from collections import Counter, defaultdict
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from pyspark.mllib.feature import Word2Vec
from parse_json import emoji_list
from string import punctuation
import numpy as np
from scipy.spatial.distance import cdist
import random

class WordPredictor(object):

    def __init__(self):
        # set up stemming agent
        # self.snowball = SnowballStemmer('english')
        # list of emoji
        self.emoji_list = emoji_list()

        # REGEX for finding emoji
        self.REGEX = u"[\U00002712\U00002714\U00002716\U0000271d\U00002721\U00002728\U00002733\U00002734\U00002744\U00002747\U0000274c\U0000274e\U00002753-\U00002755\U00002757\U00002763\U00002764\U00002795-\U00002797\U000027a1\U000027b0\U000027bf\U00002934\U00002935\U00002b05-\U00002b07\U00002b1b\U00002b1c\U00002b50\U00002b55\U00003030\U0000303d\U0001f004\U0001f0cf\U0001f170\U0001f171\U0001f17e\U0001f17f\U0001f18e\U0001f191-\U0001f19a\U0001f201\U0001f202\U0001f21a\U0001f22f\U0001f232-\U0001f23a\U0001f250\U0001f251\U0001f300-\U0001f321\U0001f324-\U0001f393\U0001f396\U0001f397\U0001f399-\U0001f39b\U0001f39e-\U0001f3f0\U0001f3f3-\U0001f3f5\U0001f3f7-\U0001f4fd\U0001f4ff-\U0001f53d\U0001f549-\U0001f54e\U0001f550-\U0001f567\U0001f56f\U0001f570\U0001f573-\U0001f579\U0001f587\U0001f58a-\U0001f58d\U0001f590\U0001f595\U0001f596\U0001f5a5\U0001f5a8\U0001f5b1\U0001f5b2\U0001f5bc\U0001f5c2-\U0001f5c4\U0001f5d1-\U0001f5d3\U0001f5dc-\U0001f5de\U0001f5e1\U0001f5e3\U0001f5ef\U0001f5f3\U0001f5fa-\U0001f64f\U0001f680-\U0001f6c5\U0001f6cb-\U0001f6d0\U0001f6e0-\U0001f6e5\U0001f6e9\U0001f6eb\U0001f6ec\U0001f6f0\U0001f6f3\U0001f910-\U0001f918\U0001f980-\U0001f984\U0001f9c0\U00003297\U00003299\U000000a9\U000000ae\U0000203c\U00002049\U00002122\U00002139\U00002194-\U00002199\U000021a9\U000021aa\U0000231a\U0000231b\U00002328\U00002388\U000023cf\U000023e9-\U000023f3\U000023f8-\U000023fa\U000024c2\U000025aa\U000025ab\U000025b6\U000025c0\U000025fb-\U000025fe\U00002600-\U00002604\U0000260e\U00002611\U00002614\U00002615\U00002618\U0000261d\U00002620\U00002622\U00002623\U00002626\U0000262a\U0000262e\U0000262f\U00002638-\U0000263a\U00002648-\U00002653\U00002660\U00002663\U00002665\U00002666\U00002668\U0000267b\U0000267f\U00002692-\U00002694\U00002696\U00002697\U00002699\U0000269b\U0000269c\U000026a0\U000026a1\U000026aa\U000026ab\U000026b0\U000026b1\U000026bd\U000026be\U000026c4\U000026c5\U000026c8\U000026ce\U000026cf\U000026d1\U000026d3\U000026d4\U000026e9\U000026ea\U000026f0-\U000026f5\U000026f7-\U000026fa\U000026fd\U00002702\U00002705\U00002708-\U0000270d\U0000270f]|[#]\U000020e3|[*]\U000020e3|[0]\U000020e3|[1]\U000020e3|[2]\U000020e3|[3]\U000020e3|[4]\U000020e3|[5]\U000020e3|[6]\U000020e3|[7]\U000020e3|[8]\U000020e3|[9]\U000020e3|\U0001f1e6[\U0001f1e8-\U0001f1ec\U0001f1ee\U0001f1f1\U0001f1f2\U0001f1f4\U0001f1f6-\U0001f1fa\U0001f1fc\U0001f1fd\U0001f1ff]|\U0001f1e7[\U0001f1e6\U0001f1e7\U0001f1e9-\U0001f1ef\U0001f1f1-\U0001f1f4\U0001f1f6-\U0001f1f9\U0001f1fb\U0001f1fc\U0001f1fe\U0001f1ff]|\U0001f1e8[\U0001f1e6\U0001f1e8\U0001f1e9\U0001f1eb-\U0001f1ee\U0001f1f0-\U0001f1f5\U0001f1f7\U0001f1fa-\U0001f1ff]|\U0001f1e9[\U0001f1ea\U0001f1ec\U0001f1ef\U0001f1f0\U0001f1f2\U0001f1f4\U0001f1ff]|\U0001f1ea[\U0001f1e6\U0001f1e8\U0001f1ea\U0001f1ec\U0001f1ed\U0001f1f7-\U0001f1fa]|\U0001f1eb[\U0001f1ee-\U0001f1f0\U0001f1f2\U0001f1f4\U0001f1f7]|\U0001f1ec[\U0001f1e6\U0001f1e7\U0001f1e9-\U0001f1ee\U0001f1f1-\U0001f1f3\U0001f1f5-\U0001f1fa\U0001f1fc\U0001f1fe]|\U0001f1ed[\U0001f1f0\U0001f1f2\U0001f1f3\U0001f1f7\U0001f1f9\U0001f1fa]|\U0001f1ee[\U0001f1e8-\U0001f1ea\U0001f1f1-\U0001f1f4\U0001f1f6-\U0001f1f9]|\U0001f1ef[\U0001f1ea\U0001f1f2\U0001f1f4\U0001f1f5]|\U0001f1f0[\U0001f1ea\U0001f1ec-\U0001f1ee\U0001f1f2\U0001f1f3\U0001f1f5\U0001f1f7\U0001f1fc\U0001f1fe\U0001f1ff]|\U0001f1f1[\U0001f1e6-\U0001f1e8\U0001f1ee\U0001f1f0\U0001f1f7-\U0001f1fb\U0001f1fe]|\U0001f1f2[\U0001f1e6\U0001f1e8-\U0001f1ed\U0001f1f0-\U0001f1ff]|\U0001f1f3[\U0001f1e6\U0001f1e8\U0001f1ea-\U0001f1ec\U0001f1ee\U0001f1f1\U0001f1f4\U0001f1f5\U0001f1f7\U0001f1fa\U0001f1ff]|\U0001f1f4\U0001f1f2|\U0001f1f5[\U0001f1e6\U0001f1ea-\U0001f1ed\U0001f1f0-\U0001f1f3\U0001f1f7-\U0001f1f9\U0001f1fc\U0001f1fe]|\U0001f1f6\U0001f1e6|\U0001f1f7[\U0001f1ea\U0001f1f4\U0001f1f8\U0001f1fa\U0001f1fc]|\U0001f1f8[\U0001f1e6-\U0001f1ea\U0001f1ec-\U0001f1f4\U0001f1f7-\U0001f1f9\U0001f1fb\U0001f1fd-\U0001f1ff]|\U0001f1f9[\U0001f1e6\U0001f1e8\U0001f1e9\U0001f1eb-\U0001f1ed\U0001f1ef-\U0001f1f4\U0001f1f7\U0001f1f9\U0001f1fb\U0001f1fc\U0001f1ff]|\U0001f1fa[\U0001f1e6\U0001f1ec\U0001f1f2\U0001f1f8\U0001f1fe\U0001f1ff]|\U0001f1fb[\U0001f1e6\U0001f1e8\U0001f1ea\U0001f1ec\U0001f1ee\U0001f1f3\U0001f1fa]|\U0001f1fc[\U0001f1eb\U0001f1f8]|\U0001f1fd\U0001f1f0|\U0001f1fe[\U0001f1ea\U0001f1f9]|\U0001f1ff[\U0001f1e6\U0001f1f2\U0001f1fc]|[0-9*#]\ufe0f\u20e3"

    def _tweet_process(self, tweet):
        """
        load json file and extract text
        """
        KEY = 'text'
        try:
            tw = json.loads(tweet.strip())
            if KEY not in tw or tw['lang']!= 'en':
                return None
            return tw

        except Exception as e:
            return None



    def _emoji_preprocess(self, tweet, predict=False):
        """
        preprocess the text which treat a emoji as a word
        also tag start and end of a setence
        """

        # add space before and after space
        for emoji in re.findall(self.REGEX, tweet):
            tweet = tweet.replace(emoji, ' ' + emoji + ' ')

        # tokenize and remove rt and @ and https://
        tweet = re.sub('\?', '', tweet)
        tweet = re.sub('\.', '', tweet)
        tweet = re.sub(',', '', tweet)
        tweet = re.sub('!', '', tweet)

        tweet_tmp = [ wd.strip(punctuation) for wd in tweet.split() \
        if not wd.startswith('@') and not wd.startswith('http') and not wd == 'rt'  ]

        if predict:
            tweet_token = ['<s>'] + tweet_tmp
        else:
            tweet_token = ['<s>'] + tweet_tmp #+ ['</s>']

        return tweet_token

    def _bigrams(self, tweet):
        """
        generate bigrams from tweets
        """
        return list(nltk.bigrams(tweet))

    def _trigrams(self, tweet):
        """
        generate trigrams from tweets
        """

        return [((w1, w2), w3) for w1, w2, w3 in nltk.trigrams(tweet)]

    def _quadgrams(self, tweet):
        """
        generate n grams
        """
        return [((w1, w2, w3), w4) for w1, w2, w3, w4 in nltk.ngrams(tweet, 4)]



    def fit(self, data=None, w_bi=1./6, w_tri=1./3, w_quad=1./2, train = False):
        """
        data: sc.textFile() object
        TODO:  save bigram, trigram, quagram dict to pickle

        """
        # set weight for n_gram models, they should add up to one
        self.w_bi = w_bi
        self.w_tri = w_tri
        self.w_quad = w_quad


        if train:
            tweets =  data\
            .filter(lambda tw: len(tw)>1)\
            .filter(lambda tw: 'created_at' in tw)\
            .map(self._tweet_process)\
            .filter(lambda tw: tw != None)\
            .map(lambda tw: tw['text'].lower() )\
            .map(self._emoji_preprocess)

            tweets.cache()

            bigram_count = tweets\
                            .flatMap(self._bigrams).map(lambda bg: (bg, 1))\
                            .reduceByKey(lambda cnt1, cnt2: cnt1+cnt2)\
                            .collect()
            trigram_count = tweets\
                            .flatMap(self._trigrams).map(lambda bg: (bg, 1))\
                            .reduceByKey(lambda cnt1, cnt2: cnt1+cnt2)\
                            .map(lambda ((key, val), cnt): ((str(key), val), cnt))\
                            .collect()
            quadgrams_count = tweets\
                            .flatMap(self._quadgrams).map(lambda bg: (bg, 1))\
                            .reduceByKey(lambda cnt1, cnt2: cnt1+cnt2)\
                            .map(lambda ((key, val), cnt): ((str(key), val), cnt))\
                            .collect()


            self.bigram_dict = defaultdict(Counter)
            self.trigram_dict = defaultdict(Counter)
            self.quadgram_dict= defaultdict(Counter)

            for ((k, w1) , cnt) in bigram_count:
                self.bigram_dict[k][w1] = cnt

            for ((k, w2), cnt) in trigram_count:
                self.trigram_dict[k][w2] = cnt

            for ((k, w3), cnt) in quadgrams_count:
                self.quadgram_dict[k][w3] = cnt

            # normalizing the Counter
            for key in self.bigram_dict:
                total = sum(self.bigram_dict[key].values())
                for val in self.bigram_dict[key]:
                    self.bigram_dict[key][val] = self.bigram_dict[key][val]/float(total)

            for key in self.trigram_dict:
                total = sum(self.trigram_dict[key].values())
                for val in self.trigram_dict[key]:
                    self.trigram_dict[key][val] = self.trigram_dict[key][val]/float(total)


            for key in self.quadgram_dict:
                total = sum(self.quadgram_dict[key].values())
                for val in self.quadgram_dict[key]:
                    self.quadgram_dict[key][val] = self.quadgram_dict[key][val]/float(total)


            self.tweets = tweets
            self._build_w2v()

        else:

            with open('data/bigram_dict.json', 'r') as f:
                self.bigram_dict = json.load(f)
            with open('data/trigram_dict.json', 'r') as f:
                self.trigram_dict = json.load(f)
            with open('data/quadgram_dict.json', 'r') as f:
                self.quadgram_dict = json.load(f)


            self.bigram_dict = defaultdict(Counter, self.bigram_dict)
            self.trigram_dict = defaultdict(Counter, self.trigram_dict)
            self.quadgram_dict = defaultdict(Counter, self.quadgram_dict)


            for key, val in self.bigram_dict.iteritems():
                self.bigram_dict[key] = Counter(val)
            for key, val in self.trigram_dict.iteritems():
                self.trigram_dict[key] = Counter(val)
            for key, val in self.quadgram_dict.iteritems():
                self.quadgram_dict[key] = Counter(val)


            self.w2v_idx = np.load('wd_idx.npy')
            self.w2v_vect = np.load('wd_vect.npy')


    def _weighted_ngram(self, key, model, wt):
        """
        redistribute probability by weight
        """
        copy_mod = model[str(key)].copy()

        for gram in copy_mod:
            copy_mod[gram] = copy_mod[gram]*wt
        return copy_mod


    def set_params(self, w_bi, w_tri, w_quad):
        """
        set weight for n_gram models, they should add up to one
        """
        self.w_bi = w_bi
        self.w_tri = w_tri
        self.w_quad = w_quad


    def _backoff_model(self, proc_str):
        """
        calculating the probability of all word frequencies from previous n_grams
        """
        bigram_mod = self._weighted_ngram(proc_str[-1:][0], self.bigram_dict, self.w_bi)
        trigram_mod = self._weighted_ngram(tuple(proc_str[-2:]), self.trigram_dict, self.w_tri)
        quadgram_mod = self._weighted_ngram(tuple(proc_str[-3:]), self.quadgram_dict, self.w_quad)
        return bigram_mod + trigram_mod + quadgram_mod

    def predict(self, string):
        """
        Perform model prediction
        string: raw string input
        w_bi, w_tri, w_quad: weights for bigram, triagram and quadgram model,
                            should add up to one
        """
        string = unicode(string)
        NUM_EMOJI_OUTPUT = 5
        # preprocess the string as you preprocess tweets
        proc_str = self._emoji_preprocess(string, predict=True)
        stupid_backoff = self._backoff_model(proc_str)

        if len(stupid_backoff) != 0 and stupid_backoff != '<s>':
            emojis = []
            words = []
            additional_emoji = []
            # find the top 5 emojis in the top frequencies
            for word in zip(*stupid_backoff.most_common())[0]:
                if word in self.emoji_list:
                    emojis.append(word)
                else:
                    words.append(word)


            if len(emojis) < NU_EMOJI_OUTPUT:
                additional_emoji = self._word_to_emoji(words[0], proc_str,NUM_EMOJI_OUTPUT - len(emojis))

            emojis += additional_emoji
            print   'Predictions:'," | ".join(emojis[:5]) +' | '+ " | ".join(words[:5])
            return " | ".join(emojis[:5]) +' | '+ " | ".join(words[:5])
        else:
            print "no word in backoff"
            print "prediction with no word in backoff",self._word_to_emoji(proc_str[-1],  proc_str[:-1], 5)
            return " | ".join([self._word_to_emoji(proc_str[-1],  proc_str[:-1], 5)] + random.sample(self.emoji_list, 4))

    def _word_to_emoji(self, wd, proc_str, n=1):
        """
        turn word into emojis by looking into similarity of predicted words.
        if predicted word not in w2v model, use the last word to find similar emojis
        """
        # try:
        #     return self._similar_word(wd)
        # except:
        #     print 'exception error'
        #     return u'\U0001f600'

        wd_vect = self.w2v_vect[self.w2v_idx == wd]
        sim_word = self.w2v_idx[cdist(wd_vect, self.w2v_vect, 'cosine').argsort().flatten()]
        if not wd_vect.any():
            wd = proc_str[-1]
            print "no word in w2v ", wd

            return self._word_to_emoji(wd, proc_str[:-1], n)
        else:
            emojis = []
            for w in sim_word:
                if w in self.emoji_list:
                    emojis.append(w)
                if len(emojis) == n:
                    return w

    def _build_w2v(self):
        """
        building word2vect model
        """
        word2vec = Word2Vec()
        self.w2v = word2vec.fit(self.tweets)

    # def _similar_word(self, wd, n=1):
    #     wd_vect = self.w2v_vect[self.w2v_idx == wd]
    #     sim_word = self.w2v_idx[cdist(wd_vect, self.w2v_vect, 'cosine').argsort().flatten()]
    #     if not wd_vect.any():
    #         print 'not such word in w2v'
    #         return u'\U0001f600'
    #     else:
    #         emojis = []
    #         for w in sim_word:
    #             if w in self.emoji_list:
    #                 emojis.append(w)
    #             if len(emojis) == n:
    #                 return w
    #
    #


    def _score(self, proc_str):
        """
        calculate the perplexity score for a single string
        """
        # proc_str = self._emoji_preprocess(string)
        perplexity = 1.0

        # check length of words
        if len(proc_str) > 4 :
            n = 4
        else:
            n = len(proc_str)

        # getting scores/probability from sentence
        k = 0
        for seg in nltk.ngrams(proc_str, n):
            k += 1
            if n == 4:
                quad = seg[:3]
                tri = seg[1:3]
                bi = seg[2:3][0]
            elif n == 3:
                quad = seg[:3]
                tri = seg[:2]
                bi = seg[1:2][0]
            elif n == 2:
                quad = seg[:3]
                tri = seg[:2]
                bi = seg[0]

            pred = seg[-1]

            perplexity *= (self.w_quad * self.quadgram_dict[str(quad)][pred]\
                            + self.w_tri * self.trigram_dict[str(tri)][pred]\
                                + self.w_bi * self.bigram_dict[bi][pred])


        return np.power(perplexity, 1./k)


    def perplexity_score(self, data):
        """
        calculate perplexity_score for a corpus
        data: rdd/ list of sentence
        """
        tweets =  data\
                .filter(lambda tw: len(tw)>1)\
                .filter(lambda tw: 'created_at' in tw)\
                .map(self._tweet_process)\
                .filter(lambda tw: tw != None)\
                .map(lambda tw: tw['text'].lower() )\
                .map(self._emoji_preprocess).collect()

        score = 0
        for tw in tweets:
            score += self._score(tw)

        return score





if __name__ == '__main__':
    # start spark instance
    # sc = SparkContext()
    # data = sc.textFile('data/twitter_dump.txt')
    WP = WordPredictor()
    WP.fit()
    # WP.fit(data)
    WP.predict('I think this is a ')
