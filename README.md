# Optimization For Sentence Classification Problem

The problem is to solve an optimization problem for a linear classifier model, logistic regression model. An alternate optimization technique is used to solve this problem. 

## Alternate Optimization Technique

The alternate optimization technique optimizes the model parameters and the word embeddings alternatively in order to improve the accuracy and get more efficient results. The alternate optimization algorithm is mentioned below step-wise:

1. Initialize the required constants for various gradient descent algorithms
2. Initialize words with their corresponding word embeddings(From Google word2vec Miklov) and the model parameters with a simple random function
3. Enter a loop : for t = 0 to T-1 (For the total number of epochs)
  a. Compute the gradient of the model parameter and apply the update
  b. Compute the gradient of the word embeddings and apply the update
  c. Compute the accuracy to check for improvement

