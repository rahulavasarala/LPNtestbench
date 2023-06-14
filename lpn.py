import numpy as np

class LPNOracle:
    def __init__(self, secret, error_rate):
        self.secret = secret
        self.dimension = len(secret)
        self.error_rate = error_rate

    def sample(self, n_amount):
        # Create random matrix.
        A = np.random.randint(0, 2, size=(n_amount, self.dimension))
        # Add Bernoulli errors.
        e = np.random.binomial(1, self.error_rate, n_amount)
        # Compute the labels.
        b = np.mod(A @ self.secret + e, 2)
        return A, b
    
p = 0.125
dim = 12
s = np.random.randint(0, 2, dim)
lpn = LPNOracle(s, p)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

dt = DecisionTreeClassifier()
#dt = LogisticRegression()

# Get 100000 samples.
A, b = lpn.sample(100000)

X_train, X_test, y_train, y_test = model_selection.train_test_split(A, b, test_size=1)

print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_test dimension= ', y_test.shape)

# Fit the tree.
dt.fit(X_train, y_train)

print(dt.predict(X_train))
print(y_train)

# Predict all canonical unit vectors (1, 0, 0, ..., 0), (0, 1, 0 ,0, ..., 0), ..., (0, 0, ..., 0, 1).
s_candidate = dt.predict(np.eye(dim))

# Check if the candidate solution is correct. This is an optimized way to check whether the solution is correct or not. 
if np.mod(A @ s_candidate + b, 2).sum() < 1400:
    print("Prediction was correct!")
    print(s_candidate, s)
else:
    print('Wrong candidate. Try again!')
    print("The Hamming weight of the vector is: {a}".format(a = np.mod(A @ s_candidate + b, 2).sum()))

print("S: {a}, S': {b}".format(a = s, b = s_candidate))