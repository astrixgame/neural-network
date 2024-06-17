class Neuron {
    constructor(numInputs) {
        this.weights = [];
        this.bias = 0.01;

        for (let i = 0; i < numInputs; i++) {
            this.weights.push(Math.random() * 2 - 1);
        }
    }

    activation(input) {
        let z = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            z += input[i] * this.weights[i];
        }
        return 1 / (1 + Math.exp(-z));
    }

    train(input, target, learningRate) {
        let output = this.activation(input);
        let error = target - output;

        for(let i = 0;i < this.weights.length;i++)
            this.weights[i] += learningRate * error * input[i];
        this.bias += learningRate * error;
    }
}

class NeuralNetwork {
    constructor(numInputs, numHiddenNeurons, numOutputs, learningRate) {
        this.hiddenLayer = [];
        this.outputLayer = [];

        // Inicializace skrytých neuronů
        for (let i = 0; i < numHiddenNeurons; i++) {
            this.hiddenLayer.push(new Neuron(numInputs));
        }

        // Inicializace výstupních neuronů
        for (let i = 0; i < numOutputs; i++) {
            this.outputLayer.push(new Neuron(numHiddenNeurons));
        }

        this.learningRate = learningRate;
    }

    // Dopředný průchod (forward pass)
    forward(input) {
        let hiddenOutputs = [];

        // Vypočet výstupů skryté vrstvy
        for (let neuron of this.hiddenLayer) {
            hiddenOutputs.push(neuron.activation(input));
        }

        let outputs = [];

        // Vypočet výstupů výstupní vrstvy
        for (let neuron of this.outputLayer) {
            outputs.push(neuron.activation(hiddenOutputs));
        }

        return outputs;
    }

    // Trénování neuronové sítě
    train(input, target) {
        // Dopředný průchod
        let hiddenOutputs = [];

        // Výpočet výstupů skryté vrstvy
        for (let neuron of this.hiddenLayer) {
            hiddenOutputs.push(neuron.activation(input));
        }

        // Výpočet výstupů výstupní vrstvy
        let outputs = [];
        for (let neuron of this.outputLayer) {
            outputs.push(neuron.activation(hiddenOutputs));
        }

        // Zpětná propagace chyby a trénování výstupní vrstvy
        for (let i = 0; i < this.outputLayer.length; i++) {
            let neuron = this.outputLayer[i];
            neuron.train(hiddenOutputs, target[i], this.learningRate);
        }

        // Trénování skryté vrstvy (pokud by byla vícevrstvá síť)

        // Návrat chyby pro účely vyhodnocení
        return outputs;
    }
}

// Inicializace neuronové sítě
let neuralNet = new NeuralNetwork(2, 10, 2, 0.1);

// Trénování sítě s vstupy [0.1, 0.5] a cílovými hodnotami [0.8, 0.3]
let inputs = [0.1, 0.5];
let targets = [0.8, 0.3];
neuralNet.train(inputs, targets);

// Dopředný průchod sítí s vstupy [0.3, 0.7]
let testInputs = [0.3, 0.7];
let outputs = neuralNet.forward(testInputs);

console.log(outputs); // Výstupní hodnoty neuronové sítě
