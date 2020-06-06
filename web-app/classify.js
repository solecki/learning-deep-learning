/*
 * Canvas drawing interface + NN classifier with pre-learned parameters
 * (weights + biases).
 */
window.onload = () => {
  // Convert learned network parameter JSON into Vector and Matrix objects.
  let w = new Array(this.weights.length);
  let b = new Array(this.biases.length);
  for (let l = 0; l < w.length; l++) {
    w[l] = new Matrix(this.weights[l].rows, this.weights[l].columns);
    b[l] = Vector.fromArray(this.biases[l].values);
    for (let r = 0; r < this.weights[l].rows; r++) {
      for (let c = 0; c < this.weights[l].columns; c++) {
        w[l].values[r][c] = this.weights[l].values[r][c];
      }
    }
  }

  let prediction = new Array(10);
  let predResult = document.getElementById('prediction');
  const canvas = document.getElementById('digits');
  const ctx = canvas.getContext('2d');
  const mouse = {x: 0, y: 0};
  const scale = 4;
  canvas.width = 28 * scale;
  canvas.height = 28 * scale;

  // Drawing settings.
  ctx.strokeStyle = "black";
  ctx.lineWidth = scale * 2;
  ctx.imageSmoothingEnabled = true;

  // Mouse events.
  canvas.addEventListener('mousemove', e => {
     mouse.x = e.pageX - canvas.offsetLeft;
     mouse.y = e.pageY - canvas.offsetTop;
  });

  canvas.addEventListener('mousedown', e => {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);

    canvas.addEventListener('mousemove', onPaint, false);
  }, false);

  // Predict digit when mouseup event fires on canvas.
  canvas.addEventListener('mouseup', () => {
    let pixValues = getPixelValues();
    prediction = predict(pixValues);
    let predVal = NeuralNetwork.indexOfMax(prediction.values)
    predResult.innerHTML = predVal;
    console.table(prediction.values);
    console.log(pixValues);
    canvas.removeEventListener('mousemove', onPaint, false);
  }, false);

  document.getElementById('reset').addEventListener('click', e => {
    resetCanvas(); 
  });

  const onPaint = () => {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
  };

  const resetCanvas = () => {
    predResult.innerHTML = '';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  // Save the image data (from canvas element) to a {@pixels} sized Vector.
  const getPixelValues = () => {
    // First create a downscaled image/canvas and read pixel values from it.
    let newCanvas = document.createElement('canvas');
    let newCanvasCtx = newCanvas.getContext('2d');
    newCanvas.id = 'newCanvas';
    newCanvas.width = canvas.width * (1 / scale);
    newCanvas.height = canvas.height * (1 / scale);
    let canvasContainer = document.getElementById('canvas');
    
    newCanvasCtx.drawImage(canvas, 0, 0, newCanvas.width, newCanvas.height);
    let img = newCanvasCtx.getImageData(0, 0, newCanvas.width, newCanvas.height);

    let mnistDigit = new Vector((img.data.length / 4));
    // Normalize RGB values (i+0|1|2 == RGB channels) to MNIST data values by
    // averaging all color channels. 
    let mnistIndex = 0;
    for (let i = 0; i < img.data.length; i += 4) {
      // i+3 == alpha channel.
      if (img.data[i+3] !== 0) {
        // Since ctx.strokeStyle == "black", the RGB-channels are not set and
        // thus we can normalize the "grayscale" value just by dividing the
        // alpha channel's value by 255.
        let avg = (img.data[i+3] / 255).toFixed(3);
        mnistDigit.values[mnistIndex] = avg;
      } else {
        mnistDigit.values[mnistIndex] = 0;
      }
      mnistIndex++;
    }

    newCanvas.remove();
    return mnistDigit;
  };

  const predict = activations => {
    for (let l = 0; l < w.length; l++) {
      activations = w[l].multiply(activations);
      activations = activations.add(b[l]);                                
      // Run each neuron through activation function.
      activations.apply(NeuralNetwork.sigmoid);
    }                                                                               

    return activations; // The output layer error.
  };
  
  /*
   * Returns the index of {@arr} with the largest value.
   */
  let indexOfMax = arr => {
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
}
