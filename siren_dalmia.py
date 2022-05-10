from typing import List
import torch
import torch.nn as nn
import numpy as np


def sine_init(m, c, w0_initial=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            return m.weight.uniform_(
                -np.sqrt(c / num_input) / w0_initial, np.sqrt(c / num_input) / w0_initial)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            return m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        """Sine activation function with w0 scaling support.

        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])

        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')


class SIREN(nn.Module):
    def __init__(self, layers: List[int], in_features: int,
                 out_features: int,
                 w0: float = 1.0,
                 w0_initial: float = 30.0,
                 bias: bool = True,
                 initializer: str = 'siren',
                 c: float = 6):
        """
        SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

        :param layers: list of number of neurons in each hidden layer
        :type layers: List[int]
        :param in_features: number of input features
        :type in_features: int
        :param out_features: number of final output features
        :type out_features: int
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        :param w0_initial: `w0` of first layer. defaults to 30 (as used in the
            paper)
        :type w0_initial: float, optional
        :param bias: whether to use bias or not. defaults to
            True
        :type bias: bool, optional
        :param initializer: specifies which initializer to use. defaults to
            'siren'
        :type initializer: str, optional
        :param c: value used to compute the bound in the siren intializer.
            defaults to 6
        :type c: float, optional

        # References:
            -   [Implicit Neural Representations with Periodic Activation
                 Functions](https://arxiv.org/abs/2006.09661)
        """
        super(SIREN, self).__init__()
        self._check_params(layers)
        # Eric: for the first layer we also make sine being sine(30*x) since we've
        # already made weight init consistent with the official
        self.layers = [nn.Linear(in_features, layers[0], bias=bias), Sine(w0=w0_initial)]

        for index in range(len(layers) - 1):
            self.layers.extend([
                nn.Linear(layers[index], layers[index + 1], bias=bias),
                Sine(w0=w0_initial)
            ])

        self.layers.append(nn.Linear(layers[-1], out_features, bias=bias))
        self.network = nn.Sequential(*self.layers)

        if initializer is not None and initializer == 'siren':
            for m in self.network.modules():
                if isinstance(m, nn.Linear) and m.weight.size(-1) == 2:
                    # Eric: first-layer init
                    print(f"initializing first layer with weight of size {m.weight.shape}")
                    first_layer_sine_init(m)
                elif isinstance(m, nn.Linear):
                    # Eric: later-layer init
                    print(f"initializing later layer with weight of size {m.weight.shape}")
                    #siren_uniform_(m.weight, mode='fan_in', c=c)
                    sine_init(m, c, w0_initial)

    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def forward(self, X):
        return self.network(X)
