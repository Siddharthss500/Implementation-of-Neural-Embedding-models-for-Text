# Optimization For Sentence Classification Problem

The problem is to optimize the model parameters and word embeddings for a sentence classification problem. An alternate optimization technique is currently implement to solve this problem. 

## Alternate Optimization Technique

The alternate optimization technique optimizes the model parameters and the word embeddings alternatively in order to improve the accuracy and get improved results. The alternate optimization algorithm is mentioned below step-wise:

1. Initialize the required constants for various gradient descent algorithms
2. Initialize words with their corresponding word embeddings(From Google word2vec Miklov) and the model parameters with a simple random function
3. Enter a loop : for t = 0 to T-1 (For the total number of epochs)
  a. Compute the gradient of the model parameter and apply the update
  b. Compute the gradient of the word embeddings and apply the update
  c. Compute the accuracy to check for improvement

## Optimization Algorithms

Five gradient descent algorithms are used along with the three gradient descent variants in the aim of optimizing the objective function to get a better trained model. The three gradient descent variants are:

1. Batch Gradient Variant
2. Stochastic Gradient Variant
3. Mini-Batch Gradient Variant

The five gradient descent algorithms that are used in this experiment are:

1. Gradient Descent Algorithm
2. Momentum Algorithm
3. Nesterov Accelerated Gradient Algorithm
4. Adaptive Gradient Algorithm
5. Root-Mean-Square Progpagation Algorithm

## Datasets used in the experiment

Three datasets are used in this experiment. These datasets are taken from Kaggle - https://www.kaggle.com/rahulin05/sentiment-labelled-sentences-data-set. Each dataset contains 1000 sentences (500 positive and 500 negative) along with their corresponding labels. The datasets are:

1. Amazon product review dataset
2. IMDB movie review dataset
3. Yelp restaurant review dataset

In the aim of assigning a representation for each word in the corpus, the word embeddings that were created by Google new data (Mikolov - https://arxiv.org/pdf/1310.4546.pdf) are used in the experiment.

Link of the dataset : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
Alternative : Found under the section, Pre-trained word and phrase vectors -> https://code.google.com/archive/p/word2vec/

# Steps to run the code

1. Download the datasets from the link(https://www.kaggle.com/rahulin05/sentiment-labelled-sentences-data-set)
2. Download the representations for words from the link(https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
3. Run the file " " using the command
If running the code in the background, mention the number of epochs to run in the place _no. of epochs_

'''
(echo no. of epochs| nohup python Sentence_Classification_With_Gradient_Descent_The_End_Alternate_Optimization_Technique.py > Sentence_Classification_Alternate_Optimize_Final_log_file.txt)&
'''

For example:
(echo 1500| nohup python Sentence_Classification_With_Gradient_Descent_The_End_Alternate_Optimization_Technique.py > Sentence_Classification_Alternate_Optimize_Final_log_file.txt)&

Otherwise, and enter the number of epochs to run

'''
python Sentence_Classification_With_Gradient_Descent_The_End_Alternate_Optimization_Technique.py
'''
