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
        except KeyError:
            id = None
            text = None
            language = None

        data.append({'id': id, 'text': text, 'language': language})

    out_dict = {'id': [], 'text': [], 'language': []}

    for row in data:
        for key, value in row.items():
            out_dict[key].append(value)

    return out_dict


def read_sent_data(sent_file):
    sent_dict = {}
    for line in sent_file:
        term, score = line.split("\t")
        sent_dict[term] = int(score)

    return sent_dict


def clean_text(text):
    if not text:
        return ''
    else:
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = re.sub('_', ' ', text)
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
        text = text.lower().strip()  # convert to lowercase and trim whitespace from beginning and end
        return text


def tokenise_text(text):
    if not text:
        return []
    else:
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 1]  # remove single-letter words
        return tokens


def preprocess_text(in_dict):
    out_dict = {'text': [], 'language': [], 'clean_text': [], 'tokens': []}

    for i in range(len(in_dict['text'])):
        text = in_dict['text'][i]
        language = in_dict['language'][i]

        clean_text_val = clean_text(text)
        tokens_val = tokenise_text(clean_text_val)

        out_dict['text'].append(text)
        out_dict['language'].append(language)
        out_dict['clean_text'].append(clean_text_val)
        out_dict['tokens'].append(tokens_val)

    return out_dict


def score_tweet(tokens, sent_dict):
    score = 0
    for token in tokens:
        if token in sent_dict:
            score += sent_dict[token]
    return score


def derive_new_sentiment(tweet_dict, sent_dict):
    # create a set of all unique terms in the tweet corpus
    tweet_vocab = set([token for tokens in tweet_dict['tokens'] for token in tokens])
    # get all the terms not in the sentiment dictionary
    new_terms = [word for word in tweet_vocab if word not in sent_dict.keys()]

    tweet_tokens = list(tweet_dict['tokens'])
    tweet_sentiment = list(tweet_dict['sentiment'])

    scores = []
    # for each new term...
    for term in new_terms:
        num_tweets = 0
        sum_sentiment = 0

        # for each tweet...
        for i, tokens in enumerate(tweet_tokens):
            # if the term is in the tweet...
            if term in tokens:
                # increase the count by 1
                num_tweets += 1
                # add the sentiment score
                sum_sentiment += tweet_sentiment[i]
        # calculate the average sentiment (this is the sentiment score of the term)
        avg_sentiment = sum_sentiment / num_tweets
        scores.append(avg_sentiment)

    return new_terms, scores


def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])

    tweet_dict = read_tweet_data(tweet_file)
    sent_dict = read_sent_data(sent_file)
    processed_tweet_dict = preprocess_text(tweet_dict)

    tweet_sent_dict = processed_tweet_dict.copy()
    tweet_sent_dict['sentiment'] = [score_tweet(tokens, sent_dict) for tokens in processed_tweet_dict['tokens']]
    new_terms, sentiment_scores = derive_new_sentiment(tweet_sent_dict, sent_dict)
    
    for result in zip(new_terms, sentiment_scores):
        print(result[0] + " " + str(result[1]))


if __name__ == '__main__':
    main()
