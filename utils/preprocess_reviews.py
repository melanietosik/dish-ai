import sys
import json
import pickle

import spacy

# Init spaCy
en_nlp = spacy.load('en')


if __name__ == "__main__":
    # Usage
    if not len(sys.argv) == 2:
        print('Usage: python {0} <Yelp dataset>'.format(sys.argv[0]))
        sys.exit(1)

    # Load and process reviews
    with open(sys.argv[1], 'r') as yelp_file:
        data = yelp_file.readlines()
    total = len(data)
    reviews = []
    for review in data:
        text = json.loads(review)['text']
        reviews.append(text)

    # Lemmatize and filter stopwords
    filtered = []
    cnt = 0

    for review in reviews:
        doc = en_nlp(review)
        text = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
        filtered.append(text)

        cnt += 1
        if cnt % 1000 == 0:
            print('{0}/{1}'.format(cnt, total), flush=True)

    pickle.dump(filtered, open('preprocessed_reviews.pickle', 'wb'))
