var inputs = [2, 4, 6, 8];
var outputs = [0, 0, 1, 1];
var weight = 0.1;
var bias = 0.01;

var learningRate = 0.1;

function sigmoid(x = 0) {
  return Math.exp(x) / (Math.exp(x) + 1)
}

function train(x) {
  for(;x > 0;x--)
    for(var i = 0;i < inputs.length;i++) {
      var output = sigmoid(inputs[i] * weight + bias);
      var eOutput = outputs[i];
      
      var error = eOutput - output;
      var grad = error * inputs[i];
      
      weight += grad * learningRate;
      bias += error * learningRate;
    }
}
train(100000);
console.log(sigmoid(5*weight+bias));
