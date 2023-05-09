from collections import Counter
import json
import re
import sys


def read_tweet_data(tweet_file):
    data = []
    for line in tweet_file:
        tweet_dict = json.loads(line)
        try:
            id = tweet_dict['id']
            text = tweet_dict['text']
            language = tweet_dict['user']['lang']
            hashtags = tweet_dict['entities']['hashtags']
        except KeyError:
            id = None
            text = None
            language = None
            hashtags = []

        data.append({'id': id, 'text': text, 'language': language, 'hashtags': hashtags})

    out_dict = {'id': [], 'text': [], 'language': [], 'hashtags': []}

    for row in data:
        for key, value in row.items():
            out_dict[key].append(value)

    return out_dict


def count_hashtags(tweet_dict):
    # extract all the hashtags
    hashtags_data = tweet_dict['hashtags']
    hashtags = [sub_dict['text'] for sub_list in hashtags_data for sub_dict in sub_list]

    counter = Counter(hashtags)
    hashtag_freq = tuple(counter.items())
    sorted_freqs = sorted(hashtag_freq, key = lambda x: x[1], reverse=True)[:10]
    
    return sorted_freqs


def write_to_file(freq):
    with open('problem_6_output.txt', 'w') as f:
        for result in freq:
            f.write(result[0] + " " + str(result[1]) + "\n")


def main():
    tweet_file = open(sys.argv[1])

    tweet_dict = read_tweet_data(tweet_file)
    hashtag_freqs = count_hashtags(tweet_dict)
    
    for result in hashtag_freqs:
        print(result[0] + " " + str(result[1]))


if __name__ == '__main__':
    main()