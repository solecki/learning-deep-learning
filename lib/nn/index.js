const fs = require('fs');
const Matrix = require('../matrix');
const Vector = require('../vector');

/**
 * Library created to experiment with common neural network features, thus
 * it contains common functionality for creating (different kinds of) neural
 * networks.
 */
class NeuralNetwork {
  /**
   * Creates a new neural network, randomizes all its weights and biases, and
   * organizes the weights and biases as matrices and vectorsi for all network
   * layers.
   * @param {Number[]} structure - An array whose length defines the
   * number of layers for the network, and each value defines the number of
   * neurons per layer. E.g., [2, 4, 8, 2] represents a four-layer network with
   * 2 input neurons, 2 hidden layers (with 4 respectively 8 neurons), and an
   * output layer with 2 neurons.
   */
  constructor(structure) {
    if (!Array.isArray(structure)) {
      console.error('Invalid function parameters. Expecting an array.');
      return undefined;
    }

    this.structure = structure;
    this.layers = structure.length;
    this.weights = Array(this.layers - 1);
    this.biases = Array(this.layers - 1);

    // Randomize weights and biases.
    for (let l = 0; l < this.weights.length; l++) {
      this.weights[l] = new Matrix(this.structure[l + 1], this.structure[l]);
      this.weights[l].randomize(-1, 1);
      this.biases[l] = new Vector(this.structure[l + 1]);
      this.biases[l].randomize(-1, 1);
    }
  }

  /**
   * Randomly splits the training data ({@examples}) into {@batchSize} sized
   * mini-batches and performs a Stochastic Gradient Descent, using each batch
   * of examples, an {@epochs} number of times.
   * @param {Object[]} examples - All the training data.
   * @param {Vector} examples[n].input - Input for example n.
   * @param {Vector} examples[n].output - Output (the target) for example n.
   * @param {Number} batchSize - The mini-batch size.
   * @param {Number} epochs - Number of epochs to train.
   * @param {Number} learningRate - Gradient descent step size.
   * @param {Object[]} [testData] - Same shape as {@examples}, but used for
   * testing the networks learning progress.
   */
  SGD(examples, batchSize, epochs, learningRate, testData) {
    let nBatches = Math.round(examples.length / batchSize);
    let miniBatches = new Array(nBatches);
    for (let e = 0; e < epochs; e++) {
      NeuralNetwork.shuffle(examples);

      // Compute the gradient for each batch of examples and update all network
      // parameters.
      let batchStart = 0;
      for (let b = 0; b < nBatches; b++) {
        this.gradientStep(examples.slice(batchStart, batchStart + batchSize), learningRate);
        batchStart += batchSize;
      }

      console.log(`Epoch ${e} just finished.`);
      if (testData) {
        this.evaluate(testData);
      }
    }

    // Save trained parameters to disk after training done.
    fs.writeFile("./weights.json", JSON.stringify(this.weights), function(err) {
      if(err) {
        return console.error(err);
    }
      console.log("Weight parameters saved to disk.");
    });
    fs.writeFile("./biases.json", JSON.stringify(this.biases), function(err) {
      if(err) {
        return console.error(err);
      }
        console.log("Bias parameters saved to disk.");
    });
  }

  /**
   * Subtracts the averaged gradient, determined by the average cost of
   * {@batch}, by {@stepSize}, while simultaneously updating the network's
   * parameters.
   * @param {Object[]} batch - The same object shape as SGD()'s {@examples}
   * parameter, but smaller.
   * @param {Number} stepSize - How much we should subtract the gradient by.
   * @returns ...
   */
  gradientStep(batch, stepSize) {
    let nablaWeights = new Array(this.layers - 1);
    let nablaBiases = new Array(this.layers - 1);
    
    for (let l = 0; l < this.layers - 1; l++) {
      nablaWeights[l] = new Matrix(this.weights[l].rows, this.weights[l].columns);
      nablaBiases[l] = new Vector(this.biases[l].size);
    }

    for (let i = 0; i < batch.length; i++) {
      let gradient = this.backpropagate(batch[i].input, batch[i].output);
      let dNablaWeights = gradient.weights;
      let dNablaBiases = gradient.biases;

      for (let i = 0; i < dNablaWeights.length; i++) {
        // Add the changes in weights and biases.
        nablaWeights[i] = dNablaWeights[i].add(nablaWeights[i]);
        nablaBiases[i] = dNablaBiases[i].add(nablaBiases[i]);
      }

      for (let i = 0; i < this.weights.length; i++) {
        // Update weights and biases by subtracting a fraction (nablaW/B *
        // (stepSize / batch.length)) of the gradient of the cost function.
        this.weights[i] = this.weights[i].
          subtract(nablaWeights[i].multiply(stepSize / batch.length));

        this.biases[i] = this.biases[i].
          subtract(nablaBiases[i].multiply(stepSize / batch.length));
      }
    }
  }

  /**
   * A vector-valued function that returns the gradient of the network's error
   * based on {@input} and {@target}.
   * @param {Vector} input.
   * @param {Vector} target.
   * @returns {Object} gradient - The gradient based on {@input} and {@target}.
   * (gradient.weights and gradient.biases are the actual Vector object
   * gradients.)
   */
  backpropagate(input, target) {
    let nablaWeights = new Array(this.layers - 1);
    let nablaBiases = new Array(this.layers - 1);
    for (let l = 0; l < this.layers - 1; l++) {
      nablaWeights[l] = new Matrix(this.weights[l].rows, this.weights[l].columns);
      nablaBiases[l] = new Vector(this.biases[l].size);
    }

    // Feed input forward and calculate all activations (sigmoid(w * a + b)) for
    // all layers, and store each layer of activations in {@zVectors}. The
    // activations will be stored in {@activations}.
    let activation = input;
    let activations = new Array(this.layers); // Input layer counts as activation layer.
    activations[0] = input;
    let zVectors = new Array(this.layers - 1);
    for (let l = 0; l < this.layers - 1; l++) {
      let z = this.weights[l].multiply(activation);
      z = z.add(this.biases[l]);
      zVectors[l] = z;

      activation = Vector.apply(NeuralNetwork.sigmoid, z);
      activations[l + 1] = activation;
    }

    // Propagate backwards.
    // For the first layer in the back prop (output layer) we need the cost
    // derivative: (pred - target) * dSigmoid. (Eq BP1.)
    let delta = NeuralNetwork.dCost(activations[this.layers - 1], target);
    let dSigmoid = Vector.apply(NeuralNetwork.dSigmoid, zVectors[this.layers - 2]);
    delta = delta.multiply(dSigmoid);

    // REMOVE: indexes clear (last nabW layer and 2nd to last asctivation layer.
    nablaBiases[this.layers - 2] = delta;
    nablaWeights[this.layers - 2] = Vector.dotProduct(delta, activations[this.layers - 2]);

    // Then calculate all the hidden layers' derivatives. (Eq BP2.)
    let revv = 0;
    for (let l = this.layers - 2; l > 0; l--) {
      let z = zVectors[l - 1]; // high to low.
      dSigmoid = Vector.apply(NeuralNetwork.dSigmoid, z);

      this.weights[l].transpose();
      delta = this.weights[l].multiply(delta); // high to low but last to first ([2, 1, 0 ...])
      this.weights[l].transpose();

      delta = delta.multiply(dSigmoid);
      nablaBiases[l - 1] = delta;
      // nablaWeights: second to last index and descending.
      // activations: third to last index and descending.
      nablaWeights[l - 1] = Vector.dotProduct(delta, activations[l - 1]);

      let gradient = {
        weights: nablaWeights,
        biases: nablaBiases,
      }

      return gradient;
    }
  }

  /**
   * Computes, layer by layer, the network's error vector given the input
   * {@activations}. Mathematical definition:
   *
   *    error = sigmoid(w * a + b)
   *
   * where {@w} is the weights matrix, {@a} are the activation vectors (layers
   * of all neurons), and {@b} are the bias vectors corresponding to each layer
   * {@a}.
   * @param {Vector} activations - The input layer.
   * @returns {Vector} activations - The errors for given input {@activations}.
   */
  feedForward(activations) {
    for (let l = 0; l < this.layers - 1; l++) {
      activations = this.weights[l].multiply(activations);
      activations = activations.add(this.biases[l]);

      // Run each neuron through activation function.
      activations.apply(NeuralNetwork.sigmoid);
    }

    return activations; // The output layer error.
  }

  /**
   * Evaluate how bad the network classifies digits based on {@testData}.
   * @returns {Number} correct - Amount of correct classifications per batch.
   */
  evaluate(testData) {
    let correct = 0;
    let predictions = new Array(testData.length);
    let targets = new Array(testData.length);
    for (let i = 0; i < testData.length; i++) {
      predictions[i] = this.feedForward(testData[i].input);
      targets[i] = testData[i].output;
    }

    for (let i = 0; i < testData.length; i++) {
      // Check if classified correctly.
      if (NeuralNetwork.indexOfMax(predictions[i].values) ===
          NeuralNetwork.indexOfMax(targets[i].values)) {
          correct++;
      }
    }

    console.log(`Successfully classified ${correct} digits out of ${testData.length}.\n`);
  }

  /**
   * @returns {Number} maxIndex - The index of {@arr} with the largest value.
   */
  static indexOfMax(arr) {
    if (arr.length === 0) {
      return -1;
    }

    let max = arr[0];
    let maxIndex = 0;

    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
        maxIndex = i;
        max = arr[i];
      }
    }
    return maxIndex;
  }

  /**
   * Sigmoid activation function.
   */
  static sigmoid(a) {
    return 1 / (1 + Math.exp(-a));
  }

  /**
   * Sigmoid derivative with respect to {@a}.
   */
  static dSigmoid(a) {
    return NeuralNetwork.sigmoid(a) * (1 - NeuralNetwork.sigmoid(a));
  }

  /**
   * Tanh activation function.
   */
  tanh(a) {
    return Math.tanh(a);
  }

  /**
   * Tanh derivative with respect to {@a}.
   */
  dTanh(a) {
    return 1 - (NeuralNetwork.tanh(x)) ** 2;
  }

  /**
   * Takes the squared difference of the output predicted by {@input} and its
   * corresponding target.
   * @returns {Vector} error.
   */
  cost(input, target) {
    if (!(input instanceof Vector) || !(target instanceof Vector)) {
      console.error('Expecting Vector objects as input.');
      return undefined;
    } else {
      console.log('we gucci');
      return 'cost vector here';
    }
  }

  /**
   * The derivative of the cost function (output - target) for
   * {@outputActivations}.
   * @param {Vector} outputActivations - Some network output.
   * @param {Vecor} target - Some labeled example's expected network output.
   * @return {Vector} - The resulting vector of the subtraction.
   */
  static dCost(outputActivations, target) {
    if (outputActivations instanceof Vector && target instanceof Vector) {
      return outputActivations.subtract(target);
    } else {
      console.error('Expecting Vector objects as input.');
      return undefined;
    }
  }

  /**
   * Shuffles an array.
   */
  static shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }
}

if (typeof module !== 'undefined') {
  module.exports = NeuralNetwork;
}
