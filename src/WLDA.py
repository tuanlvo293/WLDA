import numpy as np
from dpers import DPERm

class WLDA:
   def __init__(self):
      """
        Initialize the WLDA with default parameters.
       """
      self.priors = None
      self.mus = None
      self.cov = None
      self.classes = None
      self.n_classes = None
      self.missing_rate_feature = None

   def fit(self, X, y):
      """
        Fit the WLDA model to the data.
        
        Parameters:
        X : (numpy.ndarray) Training data, shape (n_samples, n_features)
        y : (numpy.ndarray) Class labels, shape (n_samples,)
      """
      # Get unique labels for y
      self.classes = np.unique(y)

      # Count the number of labels
      self.n_classes = len(self.classes)

      # Calculate prior probabilities for each class
      self.priors = np.array([np.sum(y == g) for g in range(self.n_classes)])/len(y)

      # Compute means for each class
      self.mus = np.array([np.nanmean(X[y == g], axis = 0) for g in range(self.n_classes)])

      # Compute covariance matrix
      self.cov = DPERm(X, y)

      # Calculate the proportion of missing values (NaNs) for each feature
      self.missing_rate_feature = np.mean(np.isnan(X), axis=0)

   def calculate_weight(self):
      """
      Calculate the weight to penalize the missing value based on missing rate of the dataset
      """
      f = lambda r : 1/(1-r)
      w = np.array([f(r)  for r in self.missing_rate_feature])
      return w

   def _weighted_missing_func(self, x):
      """
      Compute the weighted missing value for a given data point 'x' for each class.
      Parameters:
         x : (numpy.ndarray) The data point being evaluated.

      Returns:
         numpy.ndarray: covariance matrix with penalized missing data
      """
      # Calculate the inverse of covariance matrix
      pre_mat = np.linalg.inv(self.cov)
      # Create a mask: 1 - observed value , 0 - missing value
      mask = ((~np.isnan(x)).astype(int))
      # Set NaN values to zero
      x[np.isnan(x)] = 0 
      # Create a diagonal matrix by multiplying mask and the weight which is calculated from calculate_weight function
      W = np.diag(mask*self.calculate_weight())
      return np.matmul(W,np.matmul(pre_mat,W))
   
   def _discriminant_func(self, x):
      """
      Compute the discriminant function value for a given data point 'x' for each class.
      Parameters:
         x : (numpy.ndarray) The data point being evaluated.

      Returns:
         numpy.ndarray: Discriminant function values for each class.
      """
      W = self._weighted_missing_func(x)
      # Initialize an empty list to save the result
      discriminants = []
      # Loop over each class
      for g in range(self.n_classes):
         # Get the mean vector of g class
         mean_vec = self.mus[g]
         # Compute the natural logarithm of the prior probability for class 'g'
         prior = np.log(self.priors[g])
         # Compute the discriminant function value for a given data point 'x' for class 'g'
         discriminant = prior - np.matmul((x-mean_vec),np.matmul(W,(x-mean_vec).T))/2
         discriminants.append(discriminant)
      return np.array(discriminants)

   def predict(self, X):
      """
        Predict the class labels for the provided data.
        Parameters:
        X : (numpy.ndarray) Test data, shape (n_samples, n_features)

        Returns:
        numpy.ndarray: Predicted class labels, shape (n_samples,)
      """
      pred_label = np.array([])
      for x in X:
         discriminants = self._discriminant_func(x)
         pred_label = np.append(pred_label, self.classes[np.argmax(discriminants)])
      return pred_label
   
   def predict_proba(self, X):
      """
        Predict class probabilities for the provided data.

        Parameters:
        X : (numpy.ndarray) Test data, shape (n_samples, n_features).

        Returns:
        numpy.ndarray - Predicted probabilities of shape (n_samples, n_classes).
      """
      probas = []
      for x in X:
         discriminants = self._discriminant_func(x)
         exps = np.exp(discriminants - np.max(discriminants)) # For numerical stability
         # Calculate softmax probabilities
         softmax = exps/np.sum(exps)
         probas.append(softmax)

      return np.array(probas)
   
   def get_weight_covariance(self, Xtest):
      """
        Get the weighted covariance matrix for X_test

        Returns:
        numpy.ndarray - Covariance matrix
      """
      return np.mean([self._weighted_missing_func(x) for x in Xtest],axis=0)
   
   def get_means(self):
      """
        Get the mean vectors for each class.

        Returns:
        numpy.ndarray - Mean vectors for each class
      """
      return self.mus
        
