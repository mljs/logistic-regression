'use strict';

const {Matrix} = require('ml-matrix');
const LogisticRegressionTwoClasses = require('../src/logreg_2classes');
const LogisticRegression = require('../src/logreg');

describe('Logistic Regression algorithm', function () {
// test for 2 classes

    it('Test of the function used with 2 classes', function() {
        var X = new Matrix([[0, -1], [1, 0], [1, 1], [1, -1], [2, 0], [2, 1], [2, -1], [3, 2], [0, 4], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4]]);
        var Y = Matrix.columnVector([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]);

        var Xtest = new Matrix([[0, -2], [1, 0.5], [1.5, -1], [1, 4.5], [2, 3.5], [1.5, 5]]);

        var logreg = new LogisticRegressionTwoClasses({numSteps: 500000, learningRate: 5e-4});
        logreg.train(X, Y);
        var results = logreg.predict(Xtest); // compute results of the training set
        expect(results).toEqual([0, 0, 0, 1, 1, 1]);
    });
});