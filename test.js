class NeuralNetwork {
    constructor(input_nodes, hidden_nodes, output_nodes, learning_rate) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;
        this.learning_rate = learning_rate;

        // Initialize weights with random values
        this.weights_ih = this.randomMatrix(this.hidden_nodes, this.input_nodes);
        this.weights_ho = this.randomMatrix(this.output_nodes, this.hidden_nodes);
        console.log(this.weights_ho);

        // Initialize biases with random values
        this.bias_h = this.randomMatrix(this.hidden_nodes, 1);
        this.bias_o = this.randomMatrix(this.output_nodes, 1);
    }

    // Generate a matrix with random values between -1 and 1
    randomMatrix(rows, cols) {
        var matrix = [];
        for(var i = 0;i < rows;i++) {
            matrix[i] = [];
            for (var j = 0;j < cols;j++)
                matrix[i][j] = Math.random() * 2 - 1;
        }
        return matrix;
    }

    // Activation function (sigmoid)
    activation(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid function
    activationDerivative(x) {
        return x * (1 - x);
    }

    // Matrix multiplication
    multiplyMatrix(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a[0].length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // Matrix addition
    addMatrix(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    // Matrix subtraction
    subtractMatrix(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    // Element-wise multiplication
    elementwiseMultiply(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] * b[i][j];
            }
        }
        return result;
    }

    // Scalar multiplication
    scalarMultiply(a, scalar) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }

    // Transpose matrix
    transposeMatrix(matrix) {
        let result = [];
        for (let i = 0; i < matrix[0].length; i++) {
            result[i] = [];
            for (let j = 0; j < matrix.length; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }

    // Forward pass
    forward(inputs) {
        this.inputs = inputs.map(x => [x]);

        // Compute hidden layer activations
        let hidden = this.addMatrix(
            this.multiplyMatrix(this.weights_ih, this.inputs),
            this.bias_h
        );
        this.hidden = hidden.map(row => row.map(this.activation));

        // Compute output layer activations
        let output = this.addMatrix(
            this.multiplyMatrix(this.weights_ho, this.hidden),
            this.bias_o
        );
        this.output = output.map(row => row.map(this.activation));

        return this.output;
    }

    // Train the neural network
    train(inputs, targets) {
        // Forward pass
        this.forward(inputs);

        // Convert targets to matrix
        targets = targets.map(x => [x]);

        // Calculate output error
        let output_errors = this.subtractMatrix(targets, this.output);

        // Calculate hidden layer errors
        let weights_ho_t = this.transposeMatrix(this.weights_ho);
        let hidden_errors = this.multiplyMatrix(weights_ho_t, output_errors);

        // Calculate output layer gradients
        let output_gradients = this.elementwiseMultiply(
            this.output.map(row => row.map(this.activationDerivative)),
            output_errors
        );
        output_gradients = this.scalarMultiply(output_gradients, this.learning_rate);

        // Calculate hidden layer gradients
        let hidden_gradients = this.elementwiseMultiply(
            this.hidden.map(row => row.map(this.activationDerivative)),
            hidden_errors
        );
        hidden_gradients = this.scalarMultiply(hidden_gradients, this.learning_rate);

        // Update weights and biases
        let hidden_T = this.transposeMatrix(this.hidden);
        this.weights_ho = this.addMatrix(
            this.weights_ho,
            this.multiplyMatrix(output_gradients, hidden_T)
        );
        this.bias_o = this.addMatrix(this.bias_o, output_gradients);

        let inputs_T = this.transposeMatrix(this.inputs);
        this.weights_ih = this.addMatrix(
            this.weights_ih,
            this.multiplyMatrix(hidden_gradients, inputs_T)
        );
        this.bias_h = this.addMatrix(this.bias_h, hidden_gradients);
    }
}

let trainingData = [
    { inputs: [0, 0], target: [0, 0] },
    { inputs: [0, 1], target: [1, 1] },
    { inputs: [1, 0], target: [1, 0] },
    { inputs: [1, 1], target: [0, 1] }
];

let nn = new NeuralNetwork(2, 20, 2, 0.1);

for(let epoch = 0;epoch < 100000;epoch++)
    for(let data of trainingData)
        nn.train(data.inputs, data.target);


console.log("Output for [0, 0]:", nn.forward([0, 0]));
console.log("Output for [0, 1]:", nn.forward([0, 1]));
console.log("Output for [1, 0]:", nn.forward([1, 0]));
console.log("Output for [1, 1]:", nn.forward([1, 1]));