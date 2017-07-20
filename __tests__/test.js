const {Matrix} = require('ml-matrix');
const LogisticRegressionTwoClasses = require('../src/logreg_2classes');
const LogisticRegression = require('../src/logreg');

// test for 2 classes

var X = new Matrix([[0,-1], [1,0], [1,1], [1,-1], [2,0], [2,1], [2,-1], [3,2], [0,4], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4]]);
var Y = Matrix.columnVector([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]);

var Xtest = new Matrix([[0, -2], [1, 0.5], [1.5, -1], [1, 4.5], [2, 3.5], [1.5, 5]])
var Ytest = Matrix.columnVector([0, 0, 0, 1, 1, 1]);

var logreg = new LogisticRegressionTwoClasses(num_steps = 500000, learning_rate = 5e-4);
logreg.train(X, Y);
// console.log(logreg.weights); // results should be [-12.07083366   9.94427349]
var results = logreg.test(Xtest); // compute results of the training set
console.log(results);






// test for 3 classes

var X = new Matrix([[0,-1], [1,0], [1,1], [1,-1], [2,0], [2,1], [2,-1], [3,2], [0,4], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [1, 10], [1, 12], [2, 10], [2,11], [2, 14], [3, 11]]);
var Y = Matrix.columnVector([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);

var Xtest = new Matrix([[0, -2], [1, 0.5], [1.5, -1], [1, 2.5], [2, 3.5], [1.5, 4], [1, 10.5], [2.5, 10.5], [2, 11.5]])
var Ytest = Matrix.columnVector([0, 0, 0, 1, 1, 1, 2, 2, 2]);

var logreg = new LogisticRegression(num_steps = 10000000, learning_rate = 5e-5);
logreg.train(X,Y);
var finalResults = logreg.test(Xtest);
console.log(finalResults);