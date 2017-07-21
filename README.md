# logistic-regression

This is an implementation of the logistic regression. When there are more than 2 classes, the method used is the *One VS All*.

## Examples : 
```javascript
const {Matrix} = require('ml-matrix');

// our training set (X,Y)
var X = new Matrix([[0,-1], [1,0], [1,1], [1,-1], [2,0], [2,1], [2,-1], [3,2], [0,4], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [1, 10], [1, 12], [2, 10], [2,11], [2, 14], [3, 11]]);
var Y = Matrix.columnVector([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);

// the test set (Xtest, Ytest)
var Xtest = new Matrix([[0, -2], [1, 0.5], [1.5, -1], [1, 2.5], [2, 3.5], [1.5, 4], [1, 10.5], [2.5, 10.5], [2, 11.5]])
var Ytest = Matrix.columnVector([0, 0, 0, 1, 1, 1, 2, 2, 2]);

// we will train our model
var logreg = new LogisticRegression(numSteps = 10000000, learningRate = 5e-5);
logreg.train(X,Y);

// we try to predict the test set
var finalResults = logreg.predict(Xtest);
// Now, you can compare finalResults with the Ytest, which is what you wanted to have.
```
