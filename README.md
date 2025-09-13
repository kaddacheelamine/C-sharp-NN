
---

# Neural Network from Scratch in C#

This repository contains a simple implementation of a feedforward neural network built entirely from scratch in C#. It does not rely on any external machine learning libraries. The primary goal of this project is to demonstrate the fundamental concepts of neural networks, including the feedforward pass, backpropagation, and mini-batch gradient descent.

## Features

-   **Flexible Architecture**: Easily define the network structure (number of layers and neurons per layer) upon instantiation.
-   **Backpropagation Algorithm**: A full implementation of the backpropagation algorithm to compute the gradients of the cost function with respect to weights and biases.
-   **Mini-batch Gradient Descent**: The network uses mini-batch stochastic gradient descent (SGD) for efficient weight updates.
-   **Activation Functions**: Supports both **Sigmoid** and **ReLU** activation functions.
-   **Proper Weight Initialization**: Implements initialization schemes appropriate for each activation function (Xavier/Glorot for Sigmoid, He for ReLU) to ensure stable training.
-   **Self-Contained Code**: The entire implementation is in a single, well-commented file for educational purposes.

## Code Structure

The project consists of a single C# file containing a few key classes:

-   **`NNets`**: The main class representing the neural network. It encapsulates all the properties and methods needed to build, train, and test the network.
    -   `FF()`: Performs the feedforward pass.
    -   `BP()`: Executes the backpropagation algorithm to calculate gradients.
    -   `Upd()`: Updates weights and biases for a single mini-batch.
    -   `Train()`: Manages the complete training loop over epochs and batches.

-   **`MU` (Math Utilities)**: A static class containing helper mathematical functions.
    -   `RndN()`: Generates a random number from a standard normal distribution using the Box-Muller transform.
    -   `Sig()`, `SigP()`: The Sigmoid function and its derivative.
    -   `ReLU()`, `ReLUP()`: The ReLU function and its derivative.

-   **`P` (Program)**: The entry point of the application. The `Main` method handles:
    1.  Generating a synthetic dataset based on a complex boolean function.
    2.  Splitting the data into training and testing sets.
    3.  Instantiating a neural network with a `[6, 10, 1]` architecture.
    4.  Training the network on the training data.
    5.  Evaluating the network's performance on the test data and calculating its accuracy.

## How to Use

To run this project, you will need the .NET SDK installed.



2.  **Run the project:**
    You can run the code directly using the .NET CLI.
    ```bash
    dotnet run
    ```

### Expected Output

When you run the program, it will begin training the neural network. After training is complete, it will evaluate the model on the test set, printing the prediction for each sample followed by the final accuracy.

```
Training...
[1,0,0,1,1,0] => 0.372 -> 0 (target 0)
[0,1,0,0,0,1] => 0.439 -> 0 (target 1)
[1,0,0,1,1,1] => 0.422 -> 0 (target 0)
[1,0,0,1,0,0] => 0.454 -> 0 (target 0)
[0,0,0,1,1,1] => 0.407 -> 0 (target 1)
[0,1,1,1,1,1] => 0.364 -> 0 (target 1)
[1,1,0,1,1,0] => 0.294 -> 0 (target 0)
[0,0,0,0,1,0] => 0.404 -> 0 (target 1)
[1,0,0,0,1,0] => 0.420 -> 0 (target 0)
[1,0,1,0,0,0] => 0.544 -> 1 (target 1)

ACC on test = 60.00%
```
*(Note: The exact output values and accuracy may vary on each run due to the random initialization of weights and data shuffling.)*

## Example Explained

The included example trains the network to learn a complex, non-linear boolean function.

-   **Inputs**: 6 binary inputs (`a`, `b`, `c`, `d`, `e`, `f`).
-   **Output**: A single binary output determined by the function `y = ((a | b) ^ ((c & f) ^ (d | e)))`.
-   **The Challenge**: All variables used to generate the output are provided as input, making the problem fully deterministic. The network's task is to uncover the intricate, non-linear relationships between the six inputs to predict the single output. The resulting accuracy demonstrates the model's ability to approximate this complex function. Performance can be further tuned by adjusting hyperparameters like the learning rate, number of epochs, or network architecture.
-   **Network Architecture**: The network is configured with an architecture of `[6, 10, 1]` (6 input neurons, 10 hidden neurons, and 1 output neuron).

## Future Improvements

This implementation serves as a strong foundation. Potential enhancements include:

-   [ ] Add more activation functions like `Tanh` and `LeakyReLU`.
-   [ ] Implement different cost functions, such as `Cross-Entropy`.
-   [ ] Introduce regularization techniques (L1, L2, Dropout) to prevent overfitting.
-   [ ] Implement more advanced optimizers like `Adam` or `RMSprop`.
-   [ ] Improve performance by refactoring to use matrix operations (vectorization) instead of nested loops.

## License

This project is licensed under the MIT License.
