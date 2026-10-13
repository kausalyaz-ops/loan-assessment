# Network diagram

```mermaid
flowchart LR
    subgraph Input["Input layer (3 values)"]
        X1["x1"]
        X2["x2"]
        X3["x3"]
    end

    Affine1["Fully connected layer<br/>W1: 3x4<br/>b1: 4"]

    subgraph Hidden["Hidden layer (4 neurons, ReLU)"]
        H1["h1"]
        H2["h2"]
        H3["h3"]
        H4["h4"]
    end

    Affine2["Fully connected layer<br/>W2: 4x1<br/>b2: 1"]
    Output["Output neuron<br/>Sigmoid<br/>probability"]

    X1 --> Affine1
    X2 --> Affine1
    X3 --> Affine1
    Affine1 --> H1
    Affine1 --> H2
    Affine1 --> H3
    Affine1 --> H4
    H1 --> Affine2
    H2 --> Affine2
    H3 --> Affine2
    H4 --> Affine2
    Affine2 --> Output
```

The input vector has 3 values. The first fully connected layer maps 3 inputs to
4 hidden neurons using `W1` and `b1`; ReLU is applied to those 4 values. The
second layer maps the 4 hidden values to 1 output using `W2` and `b2`; Sigmoid
converts that value to an output probability.
