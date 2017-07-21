import Matrix from 'ml-matrix';

export default class LogisticRegressionTwoClasses {
    constructor(options = {}) {
        this.numSteps = options.numSteps || 500000;
        this.learningRate = options.learningRate || 5e-4;
        this.weights = options.weights ? Matrix.checkMatrix(options.weights) : null;
    }

    train(features, target) {
        var weights = Matrix.zeros(1, features.columns);

        for (var step = 0; step < this.numSteps; step++) {
            var scores = features.mmul(weights.transposeView());
            var predictions = sigmoid(scores);

            // Update weights with gradient
            var outputErrorSignal = Matrix.columnVector(predictions).neg().add(target);
            var gradient = features.transposeView().mmul(outputErrorSignal);
            weights = weights.add(gradient.mul(this.learningRate).transposeView());
        }

        this.weights = weights;
    }

    testScores(features) {
        var finalData = features.mmul(this.weights.transposeView());
        var predictions = sigmoid(finalData);
        predictions = Matrix.columnVector(predictions);
        return predictions.to1DArray();
    }

    predict(features) {
        var finalData = features.mmul(this.weights.transposeView());
        var predictions = sigmoid(finalData);
        predictions = Matrix.columnVector(predictions).round();
        return predictions.to1DArray();
    }

    static load(model) {
        return new LogisticRegressionTwoClasses(model);
    }

    toJSON() {
        return {
            numSteps: this.numSteps,
            learningRate: this.learningRate,
            weights: this.weights
        };
    }
}

function sigmoid(scores) {
    scores = scores.to1DArray();
    var result = [];
    for (var i = 0; i < scores.length; i++) {
        result.push(1 / (1 + Math.exp(-scores[i])));
    }
    return result;
}
