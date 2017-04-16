import pickle
import random
import sys

import markovify

# Ad hoc blacklist filter
adhoc_filter = []


if __name__ == "__main__":
    # Usage
    if not len(sys.argv) == 2:
        print('Usage: python {0} <preprocessed sentences .PICKLE>'.format(
            sys.argv[0]))
        sys.exit(1)

    # Load sentences
    sents = pickle.load(open(sys.argv[1], 'rb'))
    # Shuffle
    random.shuffle(sents)
    # Merge
    text = ' '.join(sents)

    # Init Markov model
    model = markovify.Text(text)

    # Random review lengths
    random_len = [2, 3, 4, 5, 6]

    while True:

        # Generated review
        markov_review = []

        # Random maximum review length
        max_len = random.choice(random_len)

        cnt = 0
        while cnt < max_len:
            sent = model.make_short_sentence(180)
            if not any(word in adhoc_filter for word in sent.split()):
                markov_review.append(sent)
                cnt += 1

        print(' '.join(markov_review))
        input('\n')
