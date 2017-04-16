# Dish AI API

Dish AI is a minimal Flask app that can be used to generate random lunch reviews. Since there's not a lot of training data on catering reviews available, I decided to add a preprocessing step as a quick and efficient workaround.

Using the public [Yelp review dataset](https://www.yelp.com/dataset_challenge), I trained a number of [topic models](https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html) and manually divided the output topics into groups of suitable and unsuitable words. For example, all topics containing only foods items (e.g. _pasta_, _rice_) would be considered useful, whereas topics around the restaurant interior (e.g. _decor_, _atmosphere_) are not useful to describe catered lunches.

After splitting each review into individual sentences, I used the word lists and a number of additional features to create a new corpus of mostly suitable review sentences. Finally, I built a [Markov chain generator](https://en.wikipedia.org/wiki/Markov_chain#Markov_text_generators) to generate new reviews based on the preprocessed training data.

## Run the app

```
git clone https://github.com/melanietosik/dish_ai
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements
APP_DEBUG=true python run.py filtered_sentences.pickle
```

Once it's up and running, go to `http://0.0.0.0:5005/generate` in your browser to generate a new review.

## Train your own

All the utility scripts for preprocessing the data and creating the Markov model are included in the `/utils` folder. Usage information is included in each file. Below is the order in which to run each file.

```
preprocess_reviews.py
topic_model.py
prepare_markov_input.py
markov_model.py
iterate_markov_input.py
```

### spaCy

I used [spaCy](https://spacy.io/) to parse the reviews. To use any of the NLP features, i.e. sentence segmentation and lemmatization, you need to download the language model for English.

```
python -m spacy download en
```

## Resources

### Topic model

The output topics for various LDA models are included in the `/topic_models` folder. The different values in the file names correspond to the number of clusters specified during training, i.e. `all_lda_500.txt` is the output of a model with 500 topics.

### Word lists

The word lists I used to filter the reviews sentences are included in the `/word_lists` folder. They might spare you any manual labeling!
