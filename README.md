# Natural Language Processing - Roadmap

## Quickstart 
### NLP Session Hands-on

- [Emoji Detection](https://colab.research.google.com/drive/1_-5IlUS5qOnJ-cuEuzOU8BpPSi8ggg16?usp=sharing): It is recommended to understand this notebook before moving into further concepts.

### Getting Started
- [IMDB Review Classfication](https://colab.research.google.com/github/markwest1972/LSTM-Example-Google-Colaboratory/blob/master/LSTM_IMDB_Sentiment_Example.ipynb): This notebook will help you to understand some basics of sentiment analysis. 

**Note:** Don't worry even if you don't understand the above notebooks. try to grasp the high level idea of the implemented mechanism. Implementation and Libraries will be understood after covering some basic concepts.

## Text Preprocessing
- Basic Libraries
  - [NLTK](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
  - [Spacy](https://www.youtube.com/watch?v=dIUTsFT2MeQ)
- Data Cleaning
  - [Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
  - [Stopwords and Punctuation removal](https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/)
- Tokenization
  - [Word and Sentence Tokenization](https://colab.research.google.com/drive/18ZnEnXKLQkkJoBXMZR2rspkWSm9EiDuZ)
  - [Subword Tokenization](https://www.tensorflow.org/text/guide/subwords_tokenizer)
  - [N-Grams](https://medium.com/@pankajchandravanshi/nlp-unlocked-n-grams-006-ceab1bc56bf4)
- Word Embedding
  - [Conceptual Understanding of Word Embedding](https://www.shanelynn.ie/get-busy-with-word-embeddings-introduction/) [Recommeded]
  - [Basic Embedding Techniques](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)
  - [Glove](https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010)
  - [Advance Embedding Techniques](https://colab.research.google.com/drive/1N7HELWImK9xCYheyozVP3C_McbiRo1nb)
- Hands-on
  - [Named Entity Recognisation using Spacy](https://colab.research.google.com/github/littlecolumns/ds4j-notebooks/blob/master/text-analysis/notebooks/Named%20Entity%20Recognition.ipynb)
  - [Text Summarization](https://colab.research.google.com/github/dipanjanS/nlp_workshop_odsc19/blob/master/Module05%20-%20NLP%20Applications/Project06%20-%20Text%20Summarization.ipynb)


## Basics of Pytorch and Tensorflow
Both Pytorch and Tensorflow are powerful libraries which is used to implement Neural Models.
- [Pytorch](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html): Follow this quickstart tutorial to get the basic understanding of pytorch.
- [Tensorflow](https://www.tensorflow.org/tutorials/quickstart/beginner) Follow this quickstart tutorial to get the basic undertanding of tensorflow 
  
**Note:** After having basic understanding of both libraries, It is recommended that you create a deeper understanding of any one of them. 
## Neural Models

- [Basics of RNN and LSTM](https://www.bouvet.no/bouvet-deler/explaining-recurrent-neural-networks): 
  - This blog explains the high level overview of RNN and LSTM architecture 
  - [IMDB Review Classfication](https://colab.research.google.com/github/markwest1972/LSTM-Example-Google-Colaboratory/blob/master/LSTM_IMDB_Sentiment_Example.ipynb) is used as an example.
  - Hands on:
    - [Text generation using Tensorflow](https://www.tensorflow.org/text/tutorials/text_generation)
    - [POS tagger using PyTorch](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
  - Interesting Ideas
    - [Fashion Mnist Classification using LSTM Network](https://www.kaggle.com/code/kmader/stacked-lstm-for-classification/notebook): Generally LSTM is used for text classification but we will use it for image classification in this notebook.
## Interesting Concepts
- **Information Retrieval**
  - [Term-Based IR](https://medium.com/@prateekgaurav/mastering-information-retrieval-building-intelligent-search-systems-46403b316109): Read the first two chapters given in this blog to get the basics of the IR.
  - Hands-on:
    - [Basic IR Implementation](https://www.kaggle.com/code/vabatista/introduction-to-information-retrieval): Follow this notebook to get a good understanding of basic IR techniques.
  - Videos:
    - [Standford CS124 Week-3](https://www.youtube.com/watch?v=kNkCfaH2rxc&list=PLaZQkZp6WhWwoDuD6pQCmgVyDbUWl_ZUi): This video explain detail explanation of above concepts.

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
- Kaggle Competitions 
  - [Disaster Tweet](https://www.kaggle.com/competitions/nlp-getting-started/overview)
  - [Contradictory, My Dear Watson](https://www.kaggle.com/code/alexia/kerasnlp-starter-notebook-contradictory-dearwatson)
  - [TREC-COVID IR](https://www.kaggle.com/code/otvioalves/trec-covid-information-retrieval)
- Readings
  - [Transformer](http://jalammar.github.io/illustrated-transformer/)
  - [POS Tagging using Advance Concept](https://www.freecodecamp.org/news/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24/)
  - [Language Modelling](https://medium.com/nlplanet/two-minutes-nlp-18-learning-resources-for-language-models-621c8680f8bb)
  
- Videos
  - [Stanford CS124](https://www.youtube.com/channel/UC_48v322owNVtORXuMeRmpA): You can follow his videos for deeper understanding.
  - [Named Entity Recognisation](https://www.youtube.com/watch?v=2XUhKpH0p4M): Identifying and classifying named entities in text
  - [Hugging Face](https://www.youtube.com/watch?v=00GKzGyWFEs&list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o): It provides large set of pre-trained models which can be used for multiple tasks
  - [Stanford - CS224N: Natural Language Processing with Deep Learning](https://www.youtube.com/channel/UC_48v322owNVtORXuMeRmpA)
- Books
  - [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
  - [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)
- Courses
  - [NLP Specialization](https://www.coursera.org/specializations/natural-language-processing)
