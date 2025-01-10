
This project proposes several supervised machine learning models with the aim of determining the
visual complexity of a sentence. The dataset contains a column for text and a column for scores, with the
scores ranging between -1 and 2. The best models for this regression task have proven to be K-NN and a
neural network built from scratch.

Visual complexity refers to the cognitive load required to process and understand textual content. This can
include factors like sentence structure, word choice, syntax, and overall readability. High visual complexity
in text might hinder comprehension, while overly simplified text might lack depth or precision, making
this balance a critical aspect in fields like education, accessibility, and content optimization. Textual visual
complexity can be influenced by several linguistic features such as: Lexical Features, Syntactic Features,
Semantic Features and also Stylistic Features.

The training dataset contains 8000 labeled samples with scores ranging from -1 to 2. The validation dataset
provides extra 500 samples with the respective scores also ranging from -1 to 2. And finally, the test dataset
contains 500 unlabeled samples.

Our preprocessing methodology emphasizes feature extraction while minimizing noise in the visual complex-
ity dataset. The pipeline consists of several key steps: text cleaning, linguistic processing, and numerical
transformation.
The initial cleaning phase removes all non-alphabetic characters and spaces, establishing a standardized
text format. We leverage spaCy’s Natural Language Processing framework for tokenization and linguistic
analysis, specifically targeting nouns, verbs, adjectives, and adverbs. This selective approach ensures the
retention of semantically significant components. To enhance data quality, we eliminate common stopwords
using NLTK’s English stopword dictionary, effectively reducing linguistic redundancy.
Text normalization is achieved through lemmatization, converting words to their root forms while
preserving semantic meaning. The final transformation utilizes TF-IDF (Term Frequency-Inverse Document
Frequency) vectorization, generating numerical feature vectors that capture word importance. Our TF-IDF
implementation extracts 5000 features, incorporating both unigrams and bigrams to capture individual word
significance and phrasal relationships.

