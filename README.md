# Optimization For Sentence Classification Problem

We consider the problem of designing attention based neural models for text applications such as sentence classification. We investigate several approaches and models to represent sentences for classification purpose. These include representing sentences using simple linear/non-linear combiner of word embeddings (with and without word embeddings optimization) and, Recurrent Neural Networks (RNNs) with and without attention mechanisms. Attention mechanism helps to focus on parts of sentences relevant for the task at hand. There are three sets of parameters: classifier model parameters, sentence representation parameters (along with word embeddings) and attention model parameters; and, optimization can be done for various combinations of these sets.

### Current Status

A linear combiner of word embeddings with equal weights is used to represent sentences and a linear classifier model is built to classify sentences. There are two implementations. In the first implementation only classifer model parameters are optimized with word embeddings fixed using Google word2vec. In the second implementation, we also optimize word embeddings initialized with Google word2vec. Model parameters and word embeddings are optimized using an alternate optimization technique. We have implemented several popular gradient descent based algorithms (e.g., momentum, RMSProp, AdaGrad - http://ruder.io/optimizing-gradient-descent/) with samples used as mini-batches. 

Implementation 1. **Model parameters only** - Sentence_Classification_Opt_Model_Param.py </br>
Implementation 2. **Model parameters and Word Embeddings** - Sentence_Classification_Opt_Mod_Param_And_Wrd_Emd.py </br>

### Next Steps

1. The alternate optimization technique is currently being tested and evaluated.
2. To build a non-linear classifier model with simple and sophisticated attention mechanism using tensorflow.

## Datasets used in the experiment

Three publicly available datasets are used for experimentation. Each dataset contains 1000 sentences (500 positive and 500 negative) along with their corresponding labels. All are binary classification problems with class labels (good,bad). The datasets are:

1. **Amazon product review dataset** - This dataset contains customer reviews for various products from the Amazon website. </br>
2. **IMDB movie review dataset** - This dataset contains customer reviews for various movies from the IMDB website. </br>
3. **Yelp restaurant review dataset** - This dataset contains reviews of customers for restaurants from the yelp website.</br>

We used word embeddings that were created by Google news data (Mikolov - https://arxiv.org/pdf/1310.4546.pdf).

Link of the dataset : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
Alternative : Found under the section, Pre-trained word and phrase vectors -> https://code.google.com/archive/p/word2vec/

# Steps to run the code

1. Download the datasets (File name: sentiment-labelled-sentences-data-set.zip) and unzip the file
2. Download the word representations from any of the links (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing, https://code.google.com/archive/p/word2vec/)
3. Run the file "**Sentence_Classification_Opt_Model_Param.py**" as instructed below:

If running the code in the background, mention the number of epochs to run in the place "*no. of epochs*"

```
(echo no. of epochs| nohup python Sentence_Classification_Opt_Model_Param.py > Sentence_Classification_Opt_Model_Param_log_file.txt)&
```
Example:
```
(echo 15| nohup python Sentence_Classification_Opt_Model_Param.py > Sentence_Classification_Opt_Model_Param_log_file.txt)&
```

Otherwise run the following command, and enter the number of epochs to run as the input
```
python Sentence_Classification_Opt_Model_Param.py
```

The other implementation can also be run in a similar manner.

# Things to do

1. Create separate codes having different functionalities such as: </br>
  -> Various Gradient Descent algorithms along with the Gradient Descent variants </br>
  -> Clean the input file and compute word representation using Google word2vec </br>
2. Modify the code to become more user friendly with controlability to the user i.e the user will be able to control the learning rate, select the algorithm to run along with a variant, modifying the input dataset and storing output, etc.

