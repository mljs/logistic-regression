'use strict';

const {Matrix} = require('ml-matrix');

class LogisticRegressionTwoClasses {

    constructor(options) {
        options = options || {};
        this.numSteps = options.numSteps || 500000;
        this.learningRate = options.learningRate || 5e-4;
        this.weights = options.weights || [];
    }

    sigmoid(scores) {
        scores = scores.to1DArray();
        var result = [];
        for (var i = 0; i < scores.length; i++) {
            result.push(1 / (1 + Math.exp(-scores[i])));
        }
        return result;
    }

    train(features, target) {
        var weights = Matrix.zeros(1, features.columns);
    
        for (var step = 0; step < this.numSteps; step++) {
            var scores = features.mmul(weights.transpose());
            var predictions = this.sigmoid(scores);

            // Update weights with gradient
            var output_error_signal = Matrix.columnVector(predictions).neg().add(target);
            var gradient = features.transpose().mmul(output_error_signal);
            weights = weights.add(gradient.mul(this.learningRate).transpose());
        }

        this.weights = weights
    }

    testScores(features) {
        var final_data = features.mmul(this.weights.transpose());
        var predictions = this.sigmoid(final_data);
        predictions = Matrix.columnVector(predictions);
        return predictions.to1DArray();
    }

    predict(features) {
        var final_data = features.mmul(this.weights.transpose());
        var predictions = this.sigmoid(final_data);
        predictions = Matrix.columnVector(predictions).round();
        return predictions.to1DArray();
    }

    load(model) {
        var newClassifier = new LogisticRegressionTwoClasses(model);
        return newClassifier;
    }

    toJSON() {
        var model = {numSteps: this.numSteps, learningRate: this.learningRate, weights: this.weights};
        return model;
    }
}

module.exports = LogisticRegressionTwoClasses;
