/**
 * Provides common matrix operations.
 */
class Matrix {
  /**
   * Creates a zero-valued {@rows} * {@columns} dimensioned matrix.
   */
  constructor(rows, columns) {
    this.rows = rows;
    this.columns = (columns !== undefined) ? columns : rows;
    this.values = Array(this.rows).fill().map(() => Array(this.columns).fill(0));
  }

  /**
   * Performs a matrix addition.
   * @returns {Matrix} res - The matrix resulting from the addition.
   */
  add(matrix) {
    if (matrix instanceof Matrix && matrix.hasDimension(this.rows, this.columns)) {
      let res = new Matrix(this.rows, this.columns);
      res.values = this.values.map((row, i) => {
        return row.map((column, j) => {
          return res.values[i][j] = row[j] + matrix.values[i][j];
        });
      });

      return res;
    } else {
      console.error('Matrix is not properly formatted.');
      return undefined;
    }
  }

  /**
   * Performs a matrix subtraction.
   * @returns {Matrix} difference - The matrix resulting from the subtraction.
   */
  subtract(matrix) {
    if (matrix instanceof Matrix && matrix.hasDimension(this.rows, this.columns)) {
      let res = new Matrix(this.rows, this.columns);
      res.values = this.values.map((row, i) => {
        return row.map((column, j) => {
          return res.values[i][j] = row[j] - matrix.values[i][j];
        });
      });
      return res;
    } else {
      console.error('Matrix is not properly formatted.');
      return undefined;
    }
  }

  /**
   * Calculates the dot product of {@v1} and {@v2}.
   * @param {Vector} v1.
   * @param {Vector} v2.
   */
  static dotProduct(v1, v2) {
    // Convert to Vector objects if ordinary arrays.
    if (Array.isArray(v1) && Array.isArray(v2)) {
      v1 = Vector.fromArray(v1);
      v2 = Vector.fromArray(v2);
    }

    if (v1 instanceof Vector && v2 instanceof Vector && v1.size === v2.size) {
      let dotProd = 0;
      v1.values.map((value, i) => dotProd += value * v2.values[i]);
      return dotProd;
    } else {
      console.error('Vectors are not properly formatted.');
      return undefined;
    }
  }

  /**
   * Perform matrix multiplication.
   * @param {Matrix|Vector|Number} matrix - The object used in the
   * multiplication.
   * @returns {Matrix|Vector} product.
   */
  multiply(matrix) {
    if (matrix instanceof Matrix && this.columns === matrix.rows) {
      let product = new Matrix(this.rows, matrix.columns);

      // Run dot product for every row-to-column relation between {@matrix} and
      // {@this}.
      this.values.map((row, i) => {
        matrix.values.map((column, j) => {
          product.values[i][j] = Matrix.dotProduct(this.values[i], column)
        });
      });

      // matrix.transpose(); // Restore transposition when done with dot products.
      return product;
    } else if(matrix instanceof Vector && this.columns === matrix.size) {
      // Multiply matrix by *vector* (n*1 matrix).
      let product = new Vector(this.rows);
      this.values.map((row, i) => {
        product.values[i] = Matrix.dotProduct(row, matrix.values);
      });

      return product;
    } else if (!isNaN(matrix)) {
      // Multiply by scalar.
      let product = new Matrix(this.rows, this.columns);
      product.values = this.values.map(row => row.map(column => column *= matrix));
      return product;
    } else {
      console.error('Matrix is not properly formatted.');
      return undefined;
    }
  }

  /**
   * Calculate the Hadamard product.
   * @returns {Matrix} product - The matrix resulting from the Hadamard operation.
   */
  hadamard(matrix) {
    if (matrix instanceof Matrix && matrix.hasDimension(this.rows, this.columns)) {
      let product = new Matrix(this.rows, this.columns);
      product.values = this.values.map((row, i) => {
        return row.map((column, j) => {
          return product.values[i][j] = row[j] * matrix.values[i][j];
        });
      });
      return product;
    } else {
      console.error('Matrix is not properly formatted.');
      return undefined;
    }
  }

  /**
   * Transposing this matrix instance.
   */
  transpose() {
    let transpose = new Matrix(this.columns, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        transpose.values[j][i] = this.values[i][j];
      }
    }

    this.rows = transpose.rows;
    this.columns = transpose.columns;
    this.values = transpose.values;
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
    if (isNaN(min) || isNaN(max) || max <= min) {
      console.error('Invalid input. Expecting two numbers: @max > @min.');
      return undefined;
    }

    let range = max - min;
    if (intValues) {
      this.values = this.values.map(row => {
        return row.map(column => Math.round(Math.random() * range + 1));
      });
    } else {
      this.values = this.values.map(row => {
        return row.map(column => Math.random() * range + min);
      });
    }
  }

  /**
   * Compare parameters to matrix dimension.
   */
  hasDimension(rows, columns) {
    return (this.rows === rows && this.columns === columns);
  }
}

if (typeof module !== 'undefined') {
  module.exports = Matrix;
}

const Vector = require('../vector');
