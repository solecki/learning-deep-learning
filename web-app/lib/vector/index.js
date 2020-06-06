//const Matrix = require('../matrix');

/**
 * Provides common vector operations.
 */
class Vector {
  /**
   * Creates a zero-valued vector. 
   */
  constructor(size) {
    this.size = size;
    this.values = Array(this.size).fill(0);
  }

  /**
   * NB: This function assumes {@v2} is as a one-row vector, even though there's
   * no transposition state for a Vector object "yet". In other words: v1's
   * columns are multiplied by {@v2}, e.g. v1.size = 10, v2.size = 30 yields a
   * 10 by 30 matrix.
   * @param {Vector} v1.
   * @param {Vector} v2.
   *
   *
   * v1 by v2 matrix returned.
   */
  static dotProduct(v1, v2) {
    // Convert to Vector objects if parameters are ordinary arrays.
    if (Array.isArray(v1) && Array.isArray(v2)) {
      v1 = Vector.fromArray(v1);
      v2 = Vector.fromArray(v2);
    }

    if (v1 instanceof Vector && v2 instanceof Vector) {
      let dotProd = new Matrix(v1.size, v2.size);
      // Take all values of v1 and multiply once by each value of v2.
      for (let i = 0; i < v1.size; i++) {
        for (let j = 0; j < v2.size; j++) {
          dotProd.values[i][j] = v1.values[i] * v2.values[j];
        }
      }
      return dotProd;
    } else {
      console.error('Vectors are not properly formatted.');
      return undefined;
    }
  }

  /**
   *
   */
  multiply(vector) {
    if(vector instanceof Vector) {
      // Vector by vector multiplication.
      let product = new Vector(this.size);
      product.values = this.values.map((value, i) => value *= vector.values[i]);
      return product;
    } else if (!isNaN(vector)) {
      // Multiply by scalar.
      let product = new Vector(this.size);
      product.values = this.values.map(value => value *= vector);
      return product;
    } else {
      console.log('Expecting vector as argument.');
      return undefined;
    }
  }

  /**
   * Performs a vector addition.
   * @returns {Vector} res - The vector resulting from the addition.
   */
  add(vector) {
    if (vector instanceof Vector && this.size === vector.size) {
      let res = new Vector(this.size);
      res.values = this.values.map((value, i) => {
          return res.values[i] = value + vector.values[i];
      });
      return res;
    } else {
      console.error('Vector is not properly formatted.');
      return undefined;
    }
  }

  /**
   * Performs a vector subtraction.
   * @returns {Vector} res - The vector resulting from the subtraction.
   */
  subtract(vector) {
    if (vector instanceof Vector && vector.size === this.size) {
      let res = new Vector(this.size);
      res.values = this.values.map((value, i) => {
        return res.values[i] = value - vector.values[i];
      });
      return res;
    } else if (!isNaN(vector)) {
      // Subtract by scalar.
      this.apply(value => value -= vector);
      return this;
    } else {
      console.error('Vector is not properly formatted.');
      return undefined;
    }
  }

  /**
   * Converts ordinary javascript array into Vector object.
   * @param {Number[]} array.
   * @returns {Vector} vector.
   */
  static fromArray(array) {
    if (Array.isArray(array)) {
      let vector = new Vector(array.length);
      for (let i = 0; i < vector.size; i++) {
        vector.values[i] = array[i];
      }
      return vector;
    } else {
      console.error('Expecting an array (Number[]) as input.');
      return undefined;
    }
  }

  /**
   * Returns a new object after applying {@func} on {vectors}'s elements.
   * @returns {Vector} res.
   */
  static apply(func, vector) {
    if (!(vector instanceof Vector)) {
      console.error('Expecting Vector object as argument.');
      return undefined;
    }

    let res = new Vector(vector.size);
    for (let i = 0; i < vector.size; i++) {
      res.values[i] = func(vector.values[i]);
    }
    return res;
  }

  /**
   * Pass function to apply for each value.
   */
  apply(func) {
    for (let i = 0; i < this.size; i++) {
      let val = this.values[i];
      this.values[i] = func(val);
    }
  }

  /**
   * Assign random values (range {@min} through {@max} inclusive) to this matrix
   * instance.
   * @param {Number} [min=0] - Minimum randomized value.
   * @param {Number} [max=1] - Maximum randomized value.
   * @param {Boolean} [intValues=false] - Only randomize integer values.
   */
  randomize(min, max, intValues) {
    min = min || 0;
    max = max || 1;
    intValues = intValues || false;
    if (isNaN(min) || isNaN(max) || max < min) {
      console.error('Invalid input. Expecting two numbers: @max > @min.');
      return undefined;
    }

    let range = max - min;
    if (intValues) {
      this.values = this.values.map(row => Math.round(Math.random() * range + 1));
    } else {
      this.values = this.values.map(row => Math.random() * range + min);
    }
  }

  /**
   * Compare parameters to matrix dimension.
   */
  hasSize(size) {
    return this.size === size;
  }
}

if (typeof module !== 'undefined') {
  module.exports = Vector;
}
