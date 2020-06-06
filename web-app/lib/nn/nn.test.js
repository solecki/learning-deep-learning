let mnist = require('mnist');
let NeuralNetwork = require('./index.js');
const Matrix = require('../matrix');
const Vector = require('../vector');
process.stdout.write('\033c');

// MNIST digit data.
let set = mnist.set(7500, 1000);
let trainingSet = set.training;
let testSet = set.test;

// Convert examples to Vector objects.
for (let i = 0; i < trainingSet.length; i++) {
  trainingSet[i].input = Vector.fromArray(trainingSet[i].input);
  trainingSet[i].output = Vector.fromArray(trainingSet[i].output);
}

for (let i = 0; i < testSet.length; i++) {
  testSet[i].input = Vector.fromArray(testSet[i].input);
  testSet[i].output = Vector.fromArray(testSet[i].output);
}

let nn = new NeuralNetwork([784, 20, 10]);

nn.SGD(trainingSet, 10, 30, 0.6, testSet);
