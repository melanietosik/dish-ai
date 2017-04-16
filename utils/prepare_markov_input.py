import json
import pickle
import random
import sys

import spacy

# Load spaCy language models
en_nlp = spacy.load('en')


if __name__ == "__main__":
    # Usage
    if not len(sys.argv) == 4:
        print('Usage: python {0} <Yelp data .JSON> <whitelist .TXT> <blacklist .TXT>'.format(
            sys.argv[0]))
        sys.exit(1)

    # Load reviews
    print('Loading reviews', flush=True)
    with open(sys.argv[1], 'r') as yelp_file:
        data = yelp_file.readlines()

    # Load whitelist and blacklist, reformat
    print('Loading wordlists', flush=True)
    # Whitelist
    whitelist = set([line.strip() for line in open(sys.argv[2], 'r')])
    with open('wordlists/whitelist.txt', 'w') as out:
        for word in sorted(whitelist):
            out.write(word + '\n')
    # Blacklist
    blacklist = set([line.strip() for line in open(sys.argv[3], 'r')])
    with open('wordlists/blacklist.txt', 'w') as out:
        for word in sorted(blacklist):
            out.write(word + '\n')

    # Reviews
    print('Parsing JSON', flush=True)
    reviews = []
    for review in data:
        text = json.loads(review)['text']
        reviews.append(text)

    # Random sample
    print('Generating random sample', flush=True)
    sample = random.sample(reviews, 1000000)

    # Entity tags
    tags = [
        'PERSON',
        'FACILTY',
        'ORG',
        'GPE',
        'LOC',
        'EVENT',
        'WORK_OF_ART',
        'LANGUAGE'
    ]

    # Useful sentences
    use_sents = []

    sample_size = len(sample)
    cnt = 0

    # Filter reviews by sentences
    print('Processing reviews', flush=True)
    for review in sample:
        doc = en_nlp(review)
        for sent in doc.sents:
            # Minimum length 3 words
            if len(sent) > 3:
                # Get all lemmas in sentence
                lemmas = set([tok.lemma_ for tok in sent])
                """
                Sentence is deemed useful if it contains

                - at least one whitelisted word
                - no blacklisted words
                - no proper nouns
                - no entities (as listed above)
                - no digits
                """
                if any(lemma in whitelist for lemma in lemmas) and \
                        not any(lemma in blacklist for lemma in lemmas) and \
                        not any(
                            tok.tag_ == 'NNP' or
                            tok.tag_ == 'NNPS' or
                            tok.ent_type_ in tags for tok in sent) and \
                        not any(char.isdigit() for char in sent.text):
                    sent_str = sent.text
                    use_sents.append(' '.join(sent_str.split()))

        cnt += 1
        if cnt % 1000 == 0:
            print('{0}/{1}'.format(cnt, sample_size), flush=True)

    # Write output
    print('Writing output', flush=True)
    total = len(use_sents)
    pickle.dump(
        use_sents,
        open('data/filtered_sents_{0}.pickle'.format(total),
             'wb')
    )
