class Neuron {
    constructor(numInputs) {
        this.numInputs = numInputs;
        this.weights = new Array(this.numInputs).fill(0.1);
        this.bias = 0.01;
    }

    train(input, error, learningRate) {
        var output = this.activation(input);
        var gradient = output * (1 - output) * error;
        for(var i = 0;i < this.numInputs;i++)
            this.weights[i] += learningRate * gradient * input[i];
        this.bias += learningRate * gradient;
    }

    activation(input) {
        var sum = this.bias;
        for(var i = 0;i < this.numInputs;i++)
            sum += input[i] * this.weights[i];
        return 1 / (1 + Math.exp(-sum));
    }
}

class NeuralNetwork {
    constructor(numInputs, numHiddenNeurons, numOutputs, learningRate) {
        this.numInputs = numInputs;
        this.numHiddenNeurons = numHiddenNeurons;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;

        this.hiddenLayers = [];
        this.outputLayers = [];

        for(var i = 0;i < this.numHiddenNeurons;i++)
            this.hiddenLayers.push(new Neuron(numInputs));

        for(var i = 0;i < this.numOutputs;i++)
            this.outputLayers.push(new Neuron(numHiddenNeurons));
    }

    forward(inputs) {
        var hiddenOutputs = [];
        var outputs = [];

        for(var neuron of this.hiddenLayers)
            hiddenOutputs.push(neuron.activation(inputs));

        for(var neuron of this.outputLayers)
            outputs.push(neuron.activation(hiddenOutputs));

        return outputs;
    }

    train(input, target) {
        var hiddenOutputs = [];
        var outputs = [];

        for(var neuron of this.hiddenLayers)
            hiddenOutputs.push(neuron.activation(input));

        for(var neuron of this.outputLayers)
            outputs.push(neuron.activation(hiddenOutputs));

        for(var i = 0;i < this.numOutputs;i++) {
            var neuron = this.outputLayers[i];
            var outputError = target[i] - outputs[i];
            neuron.train(hiddenOutputs, outputError, this.learningRate);
        }

        for(let i = 0;i < this.numHiddenNeurons;i++) {
            let neuron = this.hiddenLayers[i];
            let error = 0;
    
            for(let j = 0;j < this.numOutputs;j++) {
                let outputNeuron = this.outputLayers[j];
                let outputError = target[j] - outputNeuron.activation(hiddenOutputs);
                error += outputError * outputNeuron.weights[i];
            }
    
            neuron.train(input, error, this.learningRate);
        }
    }
}

let neuralNet = new NeuralNetwork(2, 5, 2, 0.1);

// Training the network with inputs [0.1, 0.5] and target outputs [0.8, 0.3]
let inputs = [0.1, 0.2];
let targets = [0.1, 0.2];
neuralNet.train(inputs, targets);

// Forward pass with inputs [0.3, 0.7]
let testInputs = [0.1, 0.2];
let outputs = neuralNet.forward(testInputs);

console.log(outputs); // Output values of the neural network