import sys
import pickle
import random

from gensim import corpora, models

# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html


def train_model(model_name, corpus, id2word, num_topics):
    """
    Train specified model
    """
    # LDA
    if model_name == 'lda':
        model = models.LdaModel(
            corpus,
            id2word=id2word,
            num_topics=num_topics,
            alpha='auto',
            eval_every=5,
        )
        return model
    # LSI
    elif model_name == 'lsi':
        model = models.LsiModel(
            corpus,
            id2word=id2word,
            num_topics=num_topics,
        )
        return model
    else:
        print('Invalid model name')
    return None


if __name__ == "__main__":
    # Usage
    if not len(sys.argv) == 2:
        print('Usage: python {0} <preprocessed reviews .PICKLE>'.format(
            sys.argv[0])
        )
        sys.exit(1)

    # Load preprocessed data
    print('Loading data')
    data = pickle.load(open(sys.argv[1], 'rb'))
    # Generate random sample of n reviews
    n = 100000
    sample = random.sample(data, n)

    # Assign UID to each token
    print('Generating token ID mapping')
    id2word = corpora.Dictionary(sample)  # print(dct.token2id)

    # Convert UID dictionary to BOW document-term matrix
    print('Computing BOW matrix')
    corpus = [id2word.doc2bow(review) for review in sample]

    # Define number of topics
    num_topics = [20, 50, 100, 500, 1000]
    # Specify models
    model_names = ['lda']

    for name in model_names:
        print('Starting "{0}" model'.format(name))
        for num in num_topics:

            # Generate model
            print('Training model with k={0}'.format(num))
            model = train_model(name, corpus, id2word, num)

            # Get topics and top words per topics
            results = model.show_topics(
                num_topics=-1,
                formatted=False,
                num_words=100,
            )

            # Write results to output file
            file_name = 'results/all_{0}_{1}.txt'.format(name, num)

            print('Writing output')
            with open(file_name, 'w') as out:
                for cluster in results:
                    # Get cluster index
                    cluster_id = cluster[0]
                    # Get list of most frequent words in cluster
                    top_words = [tup[0] for tup in cluster[1]]
                    # Write to file
                    out.write('#{0}'.format(cluster_id) + '\n')
                    out.write(', '.join(top_words))
                    out.write('\n\n')
