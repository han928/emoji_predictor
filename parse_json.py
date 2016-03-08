import json
import pandas as pd

######## import json to a list ###################

def parse_json(tweets_data_path='data/twitter_dump.txt'):
    # twitter dump data path

    tweets_data = []

    # parse tweets into json and store it in list
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue
    return len(tweets_data)



################ creating a list of emoji  #####################
def emoji_list():
    df_emoji = pd.read_csv('data/emoji_table.txt', encoding='utf-8', index_col=0)
    df_emoji['count']=0
    emoji_dict = df_emoji['count'].to_dict()

    df_div_emoji = pd.read_csv('data/diversity_table.txt', encoding='utf-8', index_col=0)
    df_div_emoji['count'] = 0
    div_emoji_keys = df_div_emoji['count'].to_dict().keys()
    human_emoji = list(set([ list(emoji)[0] for emoji in div_emoji_keys]))

    return emoji_dict.keys()+human_emoji



############### regex for emoji #################################
# not sure it worked on all yet
# import re
# re.findall(u'[\U0001f000-\U0001f999]', xx)


if __name__ == '__main__':
    el = emoji_list()
