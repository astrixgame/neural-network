class Neuron {
    constructor(options = {}) {
        this._numInputs = options.numInputs;
        this._weights = new Array(this._numInputs).fill(0.1);
        this._bias = 0.01;
        this._learningRate = options.learningRate || 0.1;
    }

    train(x, data) {
        for(;x > 0;x--)
            for(var i = 0;i < data.length;i++) {
                var error = data[i].output - this.predict(data[i].inputs);
                var grad = error * this._learningRate;
                for(var j = 0;j < this._numInputs;j++)
                    this._weights[i] += grad * data[i].inputs[j];
                this._bias += grad;
            }
    }

    predict(x) {
        var sum = this._bias;
        for(var i = 0;i < this._numInputs;i++)
            sum += x[i] * this._weights[i];
        return this._sigmoid(sum);
    }

    _sigmoid(x = 0) {
        return 1 / (1 + Math.exp(-x));
    }
}

var neuron = new Neuron({
    numInputs: 2
});

neuron.train(10000, [
    { inputs: [0, 0], output: 0 },
    { inputs: [1, 1], output: 0 },
    { inputs: [1, 0], output: 1 },
    { inputs: [0, 1], output: 1 },
]);

console.log((neuron.predict([0, 0])));
console.log((neuron.predict([1, 0])));
console.log((neuron.predict([0, 1])));
console.log((neuron.predict([1, 1])));