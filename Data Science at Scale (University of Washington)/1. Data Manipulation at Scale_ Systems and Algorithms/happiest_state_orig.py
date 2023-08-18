from collections import Counter, defaultdict
import json
import re
import sys

state_dict = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AS': 'American Samoa', 'AZ': 'Arizona',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut',
    'DC': 'District of Columbia', 'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia', 'GU': 'Guam',
    'HI': 'Hawaii',
    'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana',
    'KS': 'Kansas', 'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri',
    'MP': 'Northern Mariana Islands', 'MS': 'Mississippi', 'MT': 'Montana',
    'NA': 'National', 'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire',
    'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York',
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia', 'VI': 'Virgin Islands', 'VT': 'Vermont',
    'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'
}

# reverse the dictionary
state_codes = state_dict.keys()
state_values = state_dict.values()
state_dict_r = dict(zip(state_values, state_codes))


def read_tweet_data(tweet_file):
    data = []
    for line in tweet_file:
        tweet_dict = json.loads(line)
        try:
            id = tweet_dict['id']
            text = tweet_dict['text']
            language = tweet_dict['user']['lang']
            coordinates = tweet_dict['coordinates']
            geo = tweet_dict['geo']
            place = tweet_dict['place']
            location = tweet_dict['user']['location']

        except KeyError:
            id = None
            text = None
            language = None
            coordinates = None
            geo = None
            place = []
            location = None

        data.append({'id': id, 'text': text, 'language': language, 'coordinates': coordinates, 'geo': geo,
                     'place': place, 'location': location})

    out_dict = {'id': [], 'text': [], 'language': [], 'coordinates': [], 'geo': [], 'place': [], 'location': []}

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


def process_location_info(tweet_dict, state_dict, state_dict_r):
    place_data = tweet_dict['place']
    place_full_name = [sub_dict['full_name'] if sub_dict else None for sub_dict in place_data]
    state_code_init = [loc.split(",")[-1].strip().upper() if bool(loc) and re.match(r'.*, [A-Z]{2}$', loc) else None
                       for loc in place_full_name]
    state_code_init2 = [state_dict.get(code, None) for code in state_code_init]    
    state_name_init = [loc.split(",")[0].strip().title() if bool(loc) else None for loc in place_full_name]
    state_name_init2 = [loc if loc in state_dict.values() else None for loc in state_name_init]

    state_code_init3 = [state_dict_r.get(loc, None) for loc in state_name_init2]
    # collate the two
    place_state_code = [(sc1 if sc1 else sc2 if sc2 else None) for sc1, sc2 in
                        zip(state_code_init2, state_code_init3)]

    location_data = tweet_dict['location']
    state_code_init = [loc.split(",")[-1].strip().upper() if bool(loc) and re.match(r'.*, [A-Z]{2}$', loc) else None
                       for loc in location_data]
    state_code_init2 = [state_dict.get(code, None) for code in state_code_init]    
    state_name_init = [loc.split(",")[0].strip().title() if bool(loc) else None for loc in location_data]
    state_name_init2 = [loc if loc in state_dict.values() else None for loc in state_name_init]

    state_code_init3 = [state_dict_r.get(loc, None) for loc in state_name_init2]
    # combine the two
    location_state_code = [(sc1 if sc1 else sc2 if sc2 else None) for sc1, sc2 in
                           zip(state_code_init2, state_code_init3)]

    # combine the location and place data
    state_code = [(loc_sc if loc_sc else place_sc if place_sc else None) for loc_sc, place_sc
                  in zip(location_state_code, place_state_code)]
    # append to the dictionary
    tweet_dict['state_code'] = state_code

    return tweet_dict


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
    out_dict = {'text': [], 'language': [], 'clean_text': [], 'location': [], 'tokens': []}

    for i in range(len(in_dict['text'])):
        text = in_dict['text'][i]
        language = in_dict['language'][i]
        location = in_dict['state_code'][i]

        clean_text_val = clean_text(text)
        tokens_val = tokenise_text(clean_text_val)

        out_dict['text'].append(text)
        out_dict['language'].append(language)
        out_dict['location'].append(location)
        out_dict['clean_text'].append(clean_text_val)
        out_dict['tokens'].append(tokens_val)

    return out_dict


def score_tweet(tokens, sent_dict):
    score = 0
    for token in tokens:
        if token in sent_dict:
            score += sent_dict[token]
    return score


def derive_avg_sentiment(tweet_sent_dict):
    location_sentiment = defaultdict(list)
    for score, loc in zip(tweet_sent_dict['sentiment'], tweet_sent_dict['location']):
        if loc:
            location_sentiment[loc].append(score)
    avg_sentiment = {loc: sum(scores) / len(scores) for loc, scores in location_sentiment.items()}
    sorted_sentiment = sorted(avg_sentiment.items(), key = lambda x: x[1], reverse=True)

    return sorted_sentiment


def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])

    tweet_dict = read_tweet_data(tweet_file)
    sent_dict = read_sent_data(sent_file)

    tweet_loc_dict = process_location_info(tweet_dict, state_dict, state_dict_r)
    processed_tweet_dict = preprocess_text(tweet_loc_dict)

    tweet_sent_dict = processed_tweet_dict.copy()
    tweet_sent_dict['sentiment'] = [score_tweet(tokens, sent_dict) for tokens in processed_tweet_dict['tokens']]

    sentiment_by_state_code = derive_avg_sentiment(tweet_sent_dict)
    print(sentiment_by_state_code[0][0])
    

if __name__ == '__main__':
    main()
