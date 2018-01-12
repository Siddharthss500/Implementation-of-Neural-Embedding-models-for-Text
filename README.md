# Optimization For Sentence Classification Problem

The problem is to optimize the model parameters and word embeddings for a sentence classification problem. An alternate optimization technique is currently implemented which optimizes model parameters along with word embeddings. 

### Next Step

The next step is to build a non-linear classifier model in tensorflow which implements an attention based mechanism.

## Scope of the project

The optimization takes place at three stages. The three stages are :

1. **Model parameters** - (Mention the code name here) </br>
2. **Word Embeddings** - (Mention the code name here) </br>
3. **Attention based mechanism** - Future scope </br>

Currently only the first two have been implented and the codes are present above. The attention based mechanism will be implemented in tensor flow by building a non-linear model.

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
3. Run the file "**Sentence_Classification_With_Gradient_Descent_Using_Alternate_Optimization_Technique.py**" using the command

If running the code in the background, mention the number of epochs to run in the place "*no. of epochs*"

```
(echo no. of epochs| nohup python Sentence_Classification_With_Gradient_Descent_Using_Alternate_Optimization_Technique.py > Sentence_Classification_Alternate_Optimize_Final_log_file.txt)&
```
For example:
```
(echo 15| nohup python Sentence_Classification_With_Gradient_Descent_Using_Alternate_Optimization_Technique.py > Sentence_Classification_Alternate_Optimize_Final_log_file.txt)&
```

Otherwise run the following command, and enter the number of epochs to run as the input
```
python Sentence_Classification_With_Gradient_Descent_Using_Alternate_Optimization_Technique.py
```

# Things to do

1. Create separate codes having different functionalities such as: </br>
  -> Various Gradient Descent algorithms along with the Gradient Descent variants </br>
  -> Clean the input file and compute word representation using Google word2vec </br>
2. Modify the code to give more accessibility to the user i.e the user will be able to control the learning rate, select the algorithm to run along with a variant

