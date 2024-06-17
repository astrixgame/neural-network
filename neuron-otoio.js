class Neuron {
    constructor(options = {}) {
        this._weight = options.weight || 0.1;
        this._bias = options.bias || 0.01;
        this._learningRate = options.learningRate || 0.1;
    }

    train(x, data) {
        for(;x > 0;x--)
            for(var input of Object.keys(data)) {
                var error = data[input] - this.predict(input);
                this._weight += error * input * this._learningRate;
                this._bias += error * this._learningRate;
            }
    }

    predict(x) {
        return this._sigmoid(x * this._weight + this._bias);
    }

    _sigmoid(x = 0) {
        return Math.exp(x) / (Math.exp(x) + 1);
    }
}

var neuron = new Neuron();

neuron.train(100000, {
    2: 0,
    4: 0,
    6: 1,
    8: 1
});

console.log(neuron.predict(5));