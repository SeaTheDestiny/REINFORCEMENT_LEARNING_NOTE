import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities for a given state.
    
    This network implements the policy function π(a|s) that maps states to
    action probabilities using a multi-layer perceptron with ReLU activations.
    
    Parameters
    ----------
    state_size : int
        Dimension of the state space
    action_size : int
        Dimension of the action space
    hidden_size : int, optional
        Number of neurons in hidden layers (default: 128)
    
    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer
    fc2 : nn.Linear
        Second fully connected layer
    fc3 : nn.Linear
        Output layer (action logits)
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        This method initializes all linear layer weights with Xavier uniform
        initialization and biases to zero, which helps with training stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass through the actor network.
        
        Parameters
        ----------
        state : torch.Tensor
            Input state tensor of shape (batch_size, state_size)
        
        Returns
        -------
        torch.Tensor
            Action probabilities of shape (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    
class CriticNetwork(nn.Module):
    """
    Critic network that estimates Q-values for state-action pairs.
    
    This network implements the Q-function Q(s,a) that maps state-action pairs
    to their expected returns using a multi-layer perceptron with ReLU activations.
    
    Parameters
    ----------
    state_size : int
        Dimension of the state space
    action_size : int
        Dimension of the action space
    hidden_size : int, optional
        Number of neurons in hidden layers (default: 128)
    
    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer (concatenates state and action)
    fc2 : nn.Linear
        Second fully connected layer
    fc3 : nn.Linear
        Output layer (Q-value)
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        This method initializes all linear layer weights with Xavier uniform
        initialization and biases to zero, which helps with training stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        """
        Forward pass through the critic network.
        
        Parameters
        ----------
        state : torch.Tensor
            Input state tensor of shape (batch_size, state_size)
        action : torch.Tensor
            Input action tensor of shape (batch_size, action_size)
        
        Returns
        -------
        torch.Tensor
            Q-value of shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class CriticNetworkForStateValue(nn.Module):
    """
    Critic network that estimates state values V(s).
    
    This network implements the state-value function V(s) that maps states to
    their expected returns using a multi-layer perceptron with ReLU activations.
    Used in Advantage Actor-Critic (A2C) algorithms.
    
    Parameters
    ----------
    state_size : int
        Dimension of the state space
    hidden_size : int, optional
        Number of neurons in hidden layers (default: 128)
    
    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer
    fc2 : nn.Linear
        Second fully connected layer
    fc3 : nn.Linear
        Output layer (state value)
    """
    def __init__(self, state_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        This method initializes all linear layer weights with Xavier uniform
        initialization and biases to zero, which helps with training stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass through the state-value critic network.
        
        Parameters
        ----------
        state : torch.Tensor
            Input state tensor of shape (batch_size, state_size)
        
        Returns
        -------
        torch.Tensor
            State value V(s) of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)