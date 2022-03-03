import { Matrix } from 'ml-matrix';

import LogisticRegression from '../logreg.js';
import LogisticRegressionTwoClasses from '../logreg_2classes.js';

describe('Logistic Regression algorithm', () => {
  // test for 2 classes

  it('Test of the function used with 2 classes', () => {
    let X = new Matrix([
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4],
    ]);
    let Y = Matrix.columnVector([
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    ]);

    let Xtest = new Matrix([
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 4.5],
      [2, 3.5],
      [1.5, 5],
    ]);

    let logreg = new LogisticRegressionTwoClasses({
      numSteps: 500,
      learningRate: 5e-4,
    });
    logreg.train(X, Y);
    let results = logreg.predict(Xtest); // compute results of the training set
    expect(results).toStrictEqual([0, 0, 0, 1, 1, 1]);
  });

  it('Test of the prediction with 3 classes', () => {
    let X = new Matrix([
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4],
      [1, 10],
      [1, 12],
      [2, 10],
      [2, 11],
      [2, 14],
      [3, 11],
    ]);
    let Y = Matrix.columnVector([
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
    ]);

    let Xtest = new Matrix([
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 2.5],
      [2, 3.5],
      [1.5, 4],
      [1, 10.5],
      [2.5, 10.5],
      [2, 11.5],
    ]);

    let logreg = new LogisticRegression({ numSteps: 1000, learningRate: 5e-3 });
    logreg.train(X, Y);
    let finalResults = logreg.predict(Xtest);
    expect(finalResults).toStrictEqual([0, 0, 0, 1, 1, 1, 2, 2, 2]);
  });

  it('toJSON and load', () => {
    let X = new Matrix([
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4],
      [1, 10],
      [1, 12],
      [2, 10],
      [2, 11],
      [2, 14],
      [3, 11],
    ]);
    let Y = Matrix.columnVector([
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
    ]);

    let Xtest = new Matrix([
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 2.5],
      [2, 3.5],
      [1.5, 4],
      [1, 10.5],
      [2.5, 10.5],
      [2, 11.5],
    ]);

    let logreg = new LogisticRegression({ numSteps: 1000, learningRate: 5e-3 });
    logreg.train(X, Y);

    let model = JSON.parse(JSON.stringify(logreg));
    let logreg2 = LogisticRegression.load(model);
    let finalResults = logreg2.predict(Xtest);
    expect(finalResults).toStrictEqual([0, 0, 0, 1, 1, 1, 2, 2, 2]);
  });
});
