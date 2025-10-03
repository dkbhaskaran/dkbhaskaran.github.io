import torch
import torch.nn as nn

'''
torch.nn module is the cornerstone of PyTorch for building and
training of neural networks. It abstracts intricate details of
neural network operations enabling a focus on high level design.

torch.nn.Module is the base class for all the neural network
modules. It provides

1. Initialization : Using __init__ to define the layers and
   components of the network.
2. Forward Pass : Using forward to specify how data flows
   through the layers.
3. Parameter Management : Automatically tracks and optimizes
   model Parameters.

The below class defines a simple neural network with one input,
one hidden and a output layer. Essentially it is doing the
following (without torch.nn)

    w1 = torch.randn(input_dim, hidden_dim)
    b1 = torch.randn(1, hidden_dim)

    w2 = torch.randn(hidden_dim, output_dim)
    b2 = torch.randn(1, output_dim)

    def forward(x):
        hidden = torch.matmul(x, w1) + b1
        hidden = torch.relu(hidden)

        output = torch.matmul(hidden, w2) + b2
        return output
'''
class SimpleNN(nn.Module):
    def __init__ (self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()

        # nn.Linear create a fully connected layer that
        # can apply the transformation of Y = X * W + B
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    # This is called from the __call__ method
    def forward(self, input):
        # Apply hidden layer with relu activation function
        x = torch.relu(self.hidden(input))
        # Apply output layer from hidden layer input.
        return self.output(x)


model = SimpleNN(2, 4, 1)

#Example input
x = torch.tensor([1.0, 2.0])
y = model(x)
print("Output with torch.nn is ", y)

