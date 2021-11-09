# Consumer-sentiment-on-product-features
Detecting consumer's sentiment on aspects/functionalities of products from product reviews with Python.

Various aspects of a product are extracted from consumer reviews using dependency graphs and a sentiment score is assigned using the Textblob package.

# Usage

Edit the locations in config file.

Run sentiment.py. The code will ask for entering a product review statement.

Examples:

"The look of this cloth is great, it's the material that is mediocre"
"Quality of the cloth is really bad"
"Poor quality of hardware is the major issue"
"UX of this app sucks"

scrape_me_amazon.py was designed to scrap reviews from Amazon but the code is not functioning as of now.

# Requirements

python 3.6+

Textblob

NLTK

networkx

Stanford Dependency parse jar files - download from https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.8.0/

Note: This project was done many years ago and thus there are much more sophisticated ML based methods for such Feature-based sentiment analysis as of now.
