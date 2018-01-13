# Optimization For Sentence Classification Problem

We consider the problem of designing attention based neural models for text applications such as sentence classification. We investigate several approaches and models to represent sentences for classification purpose. These include representing sentences using simple linear/non-linear combiner of word embeddings (with and without word embeddings optimization) and, Recurrent Neural Networks (RNNs) with and without attention mechanisms. Attention mechanism helps to focus on parts of sentences relevant for the task at hand. There are three sets of parameters: classifier model parameters, sentence representation parameters (along with word embeddings) and attention model parameters; and, optimization can be done for various combinations of these sets.

### Current Status

A linear combiner of word embeddings with equal weights is used to represent sentences and a linear classifier model is built to classify sentences. There are two implementations. In the first implementation only classifer model parameters are optimized with word embeddings fixed using Google word2vec. In the second implementation, we also optimize word embeddings initialized with Google word2vec. Model parameters and word embeddings are optimized using an alternate optimization technique. We have implemented several popular gradient descent based algorithms (e.g., momentum, RMSProp, AdaGrad) with samples used as mini-batches. 

Implementation 1. **Model parameters only** - Sentence_Classification_Opt_Model_Param.py </br>
Implementation 2. **Model parameters and Word Embeddings** - Sentence_Classification_Opt_Mod_Param_And_Wrd_Emd.py </br>

### Next Steps

1. The alternate optimization technique is currently being tested and evaluated.
2. To build a non-linear classifier model with simple and sophisticated attention mechanism using tensorflow.

## Optimization Algorithms

Five gradient descent algorithms are used along with the three gradient descent variants in the aim of optimizing the objective function to get a better trained model. Gradient descent variants are chosen based on the sample size of the dataset that needs to be taken at a time. The three types of gradient descent variants are:

1. **Batch Gradient Variant** - In this variant, the complete dataset is taken at a time to compute the variant </br>
2. **Stochastic Gradient Variant** - In this variant, one random example from the dataset is taken to compute the variant </br>
3. **Mini-Batch Gradient Variant** - In this variant, a set of examples in the form of a mini-batch are taken at a time to compute the gradient </br>

The five gradient descent algorithms that are implemented in this experiment are:

1. Gradient Descent Algorithm
2. Momentum Algorithm
3. Nesterov Accelerated Gradient Algorithm
4. Adaptive Gradient Algorithm
5. Root-Mean-Square Propagation Algorithm

Reference for the methods/algorithms implemented - http://ruder.io/optimizing-gradient-descent/

In this code, only mini-batch gradient variant is implemented. The next update will have the implementation for the remaining two variants.

## Datasets used in the experiment

Three datasets are used in this experiment. These datasets are taken from Kaggle - https://www.kaggle.com/rahulin05/sentiment-labelled-sentences-data-set. Each dataset contains 1000 sentences (500 positive and 500 negative) along with their corresponding labels. The datasets are:

1. **Amazon product review dataset** - This dataset contains the review of customers for various products on the Amazon website which is labelled as either 1/0 i.e good/bad (binary classification problem) </br>
2. **IMDB movie review dataset** - This dataset contains the review of customers for different movies on the IMDB website which is labelled as either 1/0 i.e good/bad (binary classification problem) </br>
3. **Yelp restaurant review dataset** - This dataset contains the review of customers for different restaurants on the yelp website which is labelled as either 1/0 i.e good/bad (binary classification problem) </br>

In the aim of assigning a representation for each word in the corpus, the word embeddings that were created by Google new data (Mikolov - https://arxiv.org/pdf/1310.4546.pdf) are used in the experiment.

Link of the dataset : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
Alternative : Found under the section, Pre-trained word and phrase vectors -> https://code.google.com/archive/p/word2vec/

# Steps to run the code

1. Download the datasets from the link(https://www.kaggle.com/rahulin05/sentiment-labelled-sentences-data-set)
2. Download the representations for words from the link(https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
3. Run the file "**Sentence_Classification_Opt_Model_Param.py**" using the command

If running the code in the background, mention the number of epochs to run in the place "*no. of epochs*"

```
(echo no. of epochs| nohup python Sentence_Classification_Opt_Model_Param.py > Sentence_Classification_Opt_Model_Param_log_file.txt)&
```
For example:
```
(echo 15| nohup python Sentence_Classification_Opt_Model_Param.py > Sentence_Classification_Opt_Model_Param_log_file.txt)&
```

Otherwise run the following command, and enter the number of epochs to run as the input
```
python Sentence_Classification_Opt_Model_Param.py
```

# Things to do

1. Create separate codes having different functionalities such as: </br>
  -> Various Gradient Descent algorithms along with the Gradient Descent variants </br>
  -> Clean the input file and compute word representation using Google word2vec </br>
2. Modify the code to become more user friendly by giving more accessibility to the user i.e the user will be able to control the learning rate, select the algorithm to run along with a variant

