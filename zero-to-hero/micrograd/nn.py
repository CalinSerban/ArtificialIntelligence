"""
Lesson 3 — Build a neural network on top of YOUR engine
=======================================================

Your engine.py can differentiate any scalar expression. A neural net is just a
big scalar expression made of neurons. Here we build three classes:

    Neuron  -> one unit:  tanh(w1*x1 + w2*x2 + ... + b)
    Layer   -> a list of Neurons, all fed the same inputs
    MLP     -> a list of Layers, output of one feeding the next

Every weight and bias is a Value, so calling loss.backward() will fill in
.grad for ALL of them automatically — that's the whole point of the engine.

Fill in the TODOs, then run:  python3 train.py
"""
import random
from engine import Value


class Neuron:
    def __init__(self, n_inputs):
        # One weight per input, plus one bias. Start them random in [-1, 1].
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # x is a list of numbers (or Values), same length as self.w.
        # Compute the activation:  tanh( sum(wi * xi) + b )
        #
        # TODO: build `act` = w1*x1 + w2*x2 + ... + b, then return act.tanh()
        # Hint: start from self.b and add each self.w[i] * x[i].
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi     # (this part is given)
        return act.tanh()

    def parameters(self):
        # All the tunable Values in this neuron: the weights plus the bias.
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        # n_outputs neurons, each taking n_inputs inputs.
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # TODO: return a flat list of every parameter from every neuron.
        # Hint: loop over self.neurons and extend a list with n.parameters().
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, n_inputs, layer_sizes):
        # e.g. MLP(3, [4, 4, 1]) -> 3 inputs, two hidden layers of 4, 1 output.
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)   # output of one layer is the input to the next
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
