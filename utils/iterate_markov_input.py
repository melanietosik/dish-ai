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
        print('Usage: python {0} <preprocessed sentences .PICKLE> <whitelist .TXT> <blacklist .TXT>'.format(
            sys.argv[0]))
        sys.exit(1)

    # Load reviews
    print('Loading sentences', flush=True)
    sents = pickle.load(open(sys.argv[1], 'rb'))

    # Load whitelist and blacklist, reformat
    print('Loading wordlists', flush=True)
    whitelist = set([line.strip() for line in open(sys.argv[2], 'r')])
    with open('wordlists/whitelist.txt', 'w') as out:
        for word in sorted(whitelist):
            out.write(word + '\n')
    blacklist = set([line.strip() for line in open(sys.argv[3], 'r')])
    with open('wordlists/blacklist.txt', 'w') as out:
        for word in sorted(blacklist):
            out.write(word + '\n')

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

    sent_len = len(sents)
    cnt = 0

    # Filter sentences
    print('Processing sentences', flush=True)
    for sent in sents:

        # spaCy
        doc = en_nlp(sent)

        # Get all lemmas in sentence
        lemmas = set([tok.lemma_ for tok in doc])
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
                    tok.ent_type_ in tags for tok in doc) and \
                not any(char.isdigit() for char in doc.text):
            sent_str = doc.text
            use_sents.append(' '.join(sent_str.split()))

        cnt += 1
        if cnt % 1000 == 0:
            print('{0}/{1}'.format(cnt, sent_len), flush=True)

    # Write output
    print('Writing output', flush=True)
    total = len(use_sents)
    pickle.dump(
        use_sents,
        open('data/filtered_sents_{0}.pickle'.format(total),
             'wb')
    )
