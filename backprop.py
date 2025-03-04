# My messy NN code, no libraries, just basic stuff

# Quick sigmoid thing I made
def sigmoid(x):
    # e is like 2.71828, good enough for me
    return 1 / (1 + 2.71828 ** (-x))

# Messy sigmoid derivative, works tho
def sigmoid_deriv(x):
    return x * (1 - x)

# My inputs and stuff I want it to learn
inputs = [0.05, 0.10]  # i1, i2
targets = [0.01, 0.99]  # What I want o1, o2 to be


w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
lr = 0.5


# Just one loop for now, can crank it up later
loops = 1

# Here we go
for i in range(loops):
    # Feedforward
    h1 = w1 * inputs[0] + w2 * inputs[1] + b1
    h1_out = sigmoid(h1)

    h2 = w3 * inputs[0] + w4 * inputs[1] + b1
    h2_out = sigmoid(h2)

    o1 = w5 * h1_out + w6 * h2_out + b2
    o1_out = sigmoid(o1)

    o2 = w7 * h1_out + w8 * h2_out + b2
    o2_out = sigmoid(o2)

    # Error Rate
    err1 = 0.5 * (targets[0] - o1_out) ** 2
    err2 = 0.5 * (targets[1] - o2_out) ** 2
    total_err = err1 + err2

    # Show what I got before messing with it
    if i == 0:
        print(f"Starting error: {total_err}")
        print(f"o1 = {o1_out}, o2 = {o2_out}")

    # Backprop, time to fix it
    # Output layer tweaks
    d_o1 = (o1_out - targets[0]) * sigmoid_deriv(o1_out)
    d_o2 = (o2_out - targets[1]) * sigmoid_deriv(o2_out)

    w5 = w5 - lr * d_o1 * h1_out
    w6 = w6 - lr * d_o1 * h2_out
    w7 = w7 - lr * d_o2 * h1_out
    w8 = w8 - lr * d_o2 * h2_out
    b2 = b2 - lr * (d_o1 + d_o2)

    # Hidden layer tweaks
    d_h1 = (d_o1 * w5 + d_o2 * w7) * sigmoid_deriv(h1_out)
    d_h2 = (d_o1 * w6 + d_o2 * w8) * sigmoid_deriv(h2_out)

    w1 = w1 - lr * d_h1 * inputs[0]
    w2 = w2 - lr * d_h1 * inputs[1]
    w3 = w3 - lr * d_h2 * inputs[0]
    w4 = w4 - lr * d_h2 * inputs[1]
    b1 = b1 - lr * (d_h1 + d_h2)

    # Run it again to see if it got better
    h1 = w1 * inputs[0] + w2 * inputs[1] + b1
    h1_out = sigmoid(h1)

    h2 = w3 * inputs[0] + w4 * inputs[1] + b1
    h2_out = sigmoid(h2)

    o1 = w5 * h1_out + w6 * h2_out + b2
    o1_out = sigmoid(o1)

    o2 = w7 * h1_out + w8 * h2_out + b2
    o2_out = sigmoid(o2)

    # New error
    err1 = 0.5 * (targets[0] - o1_out) ** 2
    err2 = 0.5 * (targets[1] - o2_out) ** 2
    new_err = err1 + err2

    # Show what I ended up with
    print(f"New error: {new_err}")
    print(f"New o1 = {o1_out}, o2 = {o2_out}")
    print(f"Weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}")
    print(f"         w5={w5}, w6={w6}, w7={w7}, w8={w8}")
    print(f"Biases: b1={b1}, b2={b2}")

