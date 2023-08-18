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


def derive_term_frequency(processed_dict):
    # create the tweet corpus
    corpus = [token for tokens in processed_dict['tokens'] for token in tokens]

    counter = Counter(corpus)
    term_freq = tuple(counter.items())
    term_freq_prop = {term: count / len(corpus) for term, count in term_freq}

    return term_freq


def main():
    tweet_file = open(sys.argv[1])

    tweet_dict = read_tweet_data(tweet_file)
    processed_dict = preprocess_text(tweet_dict)
    term_freq = derive_term_frequency(processed_dict)
    
    for result in term_freq:
        print(result[0] + " " + str(result[1]))


if __name__ == '__main__':
    main()
