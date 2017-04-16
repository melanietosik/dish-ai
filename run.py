import json
import pickle
import random
import sys

import markovify

from flask import (
    Flask,
)

# Create instance of Flask class
app = Flask(__name__, static_url_path='')

# Ad hoc blacklist filter
adhoc_filter = []


@app.route('/generate')
def generate():
    """
    Generate random lunch review
    """
    # Random review lengths
    random_len = [3, 4, 5, 6]

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

    return json.dumps(
        {
            'review': ' '.join(markov_review),
        }
    )


if __name__ == '__main__':
    # Usage
    if len(sys.argv) == 2:
        # Load preprocessed sentences
        sents = pickle.load(open(sys.argv[1], 'rb'))
        # Shuffle
        random.shuffle(sents)
        # Merge
        text = ' '.join(sents)
        # Init Markov model
        model = markovify.Text(text)
    else:
        print('Usage: python {0} <preprocessed sentences .PICKLE>'.format(
            sys.argv[0]))
        sys.exit(1)
    # Set port
    port = 5005
    # Set debug flag
    app.debug = True
    # Make API externally visible
    app.run(host='0.0.0.0', port=port)
