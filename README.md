# Weighted missing Linear Discriminant Analysis (WLDA)
This is a classifier package for LDA under missing data, which can be easily installed with pip.
The code repository associated with the paper: "Weighted Missing Linear Discriminant Analysis: An Explainable Approach for Classification with Missing Data". This paper is under evaluation for the journal you can find it at https://arxiv.org/pdf/2407.00710
# 1. Introduction
Linear Discriminant Analysis (LDA) remains a popular choice for classification due to its interpretable nature, derived from its capacity to model class distributions and enhance class separation through linear combinations of features. However, real-world datasets often suffer from incomplete data, posing substantial challenges for both classification accuracy and model interpretability. In this paper, we introduce a novel and robust classification method, **termed Weighted missing Linear Discriminant Analysis (WLDA)**, which extends LDA to handle datasets with missing values without the need for imputation. Our approach innovatively incorporates a weight matrix that penalizes missing entries, thereby refining parameter estimation directly on incomplete data. Experimental results across various datasets demonstrate that WLDA consistently outperforms traditional methods, especially in challenging environments where missing values are prevalent in both training and test datasets. This advancement provides a critical tool for improving classification accuracy and maintaining model transparency in the face of incomplete data.
# 2. About this repository
This repository is structured as follows
```bash
.
├── README.md
├── example.ipynb
├── setup.txt
└── src
    ├── WLDA.py
    ├── __init__.py
    ├── functions.py
    └── dpers.py
```
Here, in `/src/` folder:
- `functions` implements some algorithms needed, like generating missing data and regularization.
- `dpers` implements the DPER algorithm for multiple class data for estimating the covariance matrix used in WLDA.
- `WLDA` implements the WLDA algorithm for classification.
# 3. Usages
## 3.1. Installation
Install the package with
```bash
!pip install git+https://github.com/tuanlvo293/WLDA.git 
```
## 3.2. Example
Assume we have data (X,y) where X is continuous data and y is a class. Here, X may contain missing values. In this example, we generate randomly missing values and split the data into train and test sets with a ratio 7:3. Please note that X,y is a type of numpy.array. The details can be found in the file example.ipynb (https://github.com/tuanlvo293/WLDA/blob/main/example.ipynb)
```bash
from WLDA import WLDA

# Here we generate missing values
Xm = generate_randomly_missing(X,missing_rate)

# Train_test_split with test_size = 30%
X_train, X_test, y_train, y_test = train_test_split(Xm, y, test_size = 0.3, random_state = 0)

# Create a classifier
model = WLDA()

# Fit the classifier on the training set 
model.fit(X_train, y_train)

# Apply the classifier to the test set which may contain missing values
y_predict = model.predict(X_test).flatten()

# The accuracy of classification
accuracy = accuracy_score(y_test, y_predict)
```
# 4. Comparison

## References and contact
We recommend you cite our following paper when using these codes for further investigation:
```bash
@article{vo2024weighted,
  title={Weighted Missing Linear Discriminant Analysis: An Explainable Approach for Classification with Missing Data},
  author={Vo, Tuan L and Dang, Uyen and Nguyen, Thu},
  journal={arXiv preprint arXiv:2407.00710},
  year={2024}
}
```
You can just send additional requests directly to the corresponding author Tuan L. Vo (tuanlvo293@gmail.com) for the appropriate permission.
