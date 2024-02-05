# NLP - Roadmap

- [NLP - Roadmap](#nlp---roadmap)
  - [Pre-requisites](#pre-requisites)
  - [NLP Roadmap](#nlp-roadmap)
    - [Text Preprocessing](#text-preprocessing)
    - [Part of Speech Tagging](#part-of-speech-tagging)
    - [Named Entity Recognition](#named-entity-recognition)
    - [Text Classification](#text-classification)
    - [Sentiment Analysis](#sentiment-analysis)
    - [Language Modelling](#language-modelling)
  - [Datasets to Work On](#datasets-to-work-on)
  - [Other Resources](#other-resources)
  - [Advanced Resources](#advanced-resources)

## Pre-requisites

- **Python**
  - Notebooks From Tinkerers' Lab Tutorial
    - [Basics of Python](https://colab.research.google.com/drive/1xxJ1qIWJ_5SecKFm3dm2yTIdbdMq-xtva)
    - [Numpy](https://colab.research.google.com/drive/128UOdam4NvP-pCihfRCqDPJrvuwfRgR0)
    - [Pandas](https://colab.research.google.com/drive/1bvIhERBGq5Mnx_bpsFOeicfleiJKeOpW)
    - [Matplotlib](https://colab.research.google.com/drive/1OhF2anGzdWr5QhZgu__JvrOWQdblNFhp)
    - [Seaborn](https://colab.research.google.com/drive/1OX-UZBRfyWB7rUIflBZxTY7xCNc2Yj5H)
  - [Freecodecamp](https://www.youtube.com/watch?v=LHBE6Q9XlzI)
- **Mathematics** - Suggested to have a good understanding of the following topics as it helps in every aspect of Machine Learning and Deep Learning
  - Linear Algebra
  - Probability and Statistics
  - Calculus
  - Suggested course: [Coursera - Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning)
  - [Probability and Statistics](https://www.khanacademy.org/math/statistics-probability)
  - [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **Fundamentals of Machine Learning**
  - Basic understanding of **supervised** and **unsupervised learning** and **evaluation metrics**
  - Familiarity with libraries like `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
  - [Machine Learning By Andrew Ng](https://www.coursera.org/learn/machine-learning)
  - [Machine learning YT](https://www.youtube.com/watch?v=i_LwzRVP7bg)
- **Fundamentals of Deep Learning**
  - Understanding of **neural networks**, **backpropagation**, **activation functions**, **optimizers**, **loss functions**
  - Familiarity with libraries like (`tensorflow`, `keras`) or `pytorch`
  - [Coursera - Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
  - [Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c)

## NLP Roadmap

### Text Preprocessing

- Techniques for cleaning and preparing text data
- First step is to learn about techniques like **Tokenization**, **Lemmatization** , **Stemming** **Parts of Speech** , **Stopwords removal** and **Punctuation removal**
- Second step is learning about **Bag of Words**, **Term Frequency-Inverse Document Frequency** , **Unigram, Bigram and Ngrams**
- Finally we learn how to convert text data into numerical data using **Word Embeddings** like **Word2Vec**, **GloVe**, **FastText** and **BERT**
- [Medium: Step1](https://medium.com/@maleeshadesilva21/preprocessing-steps-for-natural-language-processing-nlp-a-beginners-guide-d6d9bf7689c9)
- [BoW, TF-IDF, N-grams](https://medium.com/analytics-vidhya/fundamentals-of-bag-of-words-and-tf-idf-9846d301ff22)
- Selected Videos of [Deep Learning By codebasics](https://www.youtube.com/watch?v=Mubj_fqiAv8&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO)

### Part of Speech Tagging

- Understanding the different parts of speech and how to tag them
- [Freecodecamp Blogpost](https://www.freecodecamp.org/news/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24/)

### Named Entity Recognition

- Identifying and classifying named entities in text
- [NER](https://www.youtube.com/watch?v=2XUhKpH0p4M)

### Text Classification

- Classifying text into different categories, for example spam detection as well as sentiment analysis

### Sentiment Analysis

- Understanding the sentiment of a given text

### Language Modelling

- A Language Model is a statistical model that is able to predict the next word in the sequence given the words that precede it
- [Medium: Blog Post](https://medium.com/nlplanet/two-minutes-nlp-18-learning-resources-for-language-models-621c8680f8bb)

## Datasets to Work On

- **Sentiment Analysis**
  - [IMDB Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  - [Amazon Product Reviews](https://www.kaggle.com/bittlingmayer/amazonreviews)
- **Text Classification**
  - [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
- **Named Entity Recognition**
  - [Kaggle - NER](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
- **Language Modelling**
  - [Wikipedia Text](https://www.kaggle.com/mikeortman/wikipedia-sentences)
  - [Shakespeare Text](https://www.kaggle.com/kingburrito666/shakespeare-plays)
- **Question Answering**
  - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
  - [Kaggle - Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering)
- **Machine Translation**
  - [Kaggle - Guide](https://www.kaggle.com/code/kkhandekar/machine-translation-beginner-s-guide)
  - [Kaggle - Seq2Seq](https://www.kaggle.com/code/harshjain123/machine-translation-seq2seq-lstms)
  - [Kaggle - Eng2French](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset)
- **Text Summarization**
  - [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)
  - [BBC News Summary](https://www.kaggle.com/pariza/bbc-news-summary)
- **Chatbot**
  - [Cornell Movie Dialogs](https://www.kaggle.com/Cornell-University/movie-dialog-corpus)
  - [Twitter Chatbot](https://www.kaggle.com/kausr25/twitter-chatbots)
  - [Kaggle - Chatbot](https://www.kaggle.com/kausr25/chatterbotenglish)
- **Text Generation**
  - [Shakespeare Text](https://www.kaggle.com/kingburrito666/shakespeare-plays)
  - [Trump Tweets](https://www.kaggle.com/austinreese/trump-tweets)
- **Text to Speech**
  - [Common Voice](https://www.kaggle.com/mozillaorg/common-voice)
- **Speech to Text**
  - [Kaggle - Speech to Text](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)

## Other Resources

- [NLTK](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
- [Spacy](https://www.youtube.com/watch?v=dIUTsFT2MeQ)
- [Tensorflow](https://www.youtube.com/watch?v=tPYj3fFJGjk)
- [Tensorflow for NLP](https://www.youtube.com/watch?v=B2q5cRJvqI8)
- [Hugging Face](https://www.youtube.com/watch?v=00GKzGyWFEs&list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o)
- [Kaggle](https://www.kaggle.com)
- Some other roadmaps and list of resources:
  - [NLP Roadmap](https://github.com/pemagrg1/Natural-Language-Processing-NLP-Roadmap)
  - [Medium Roadmap](https://aqsazafar81.medium.com/natural-language-processing-roadmap-step-by-step-guide-5fbfcc61f9d9)
  - [Medium Roadmap2](https://medium.com/aimonks/roadmap-to-learn-natural-language-processing-in-2023-6e3a9372b8cc)

## Advanced Resources

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
- [Stanford - CS224N: Natural Language Processing with Deep Learning](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)
