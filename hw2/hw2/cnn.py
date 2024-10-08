import torch
import torch.nn as nn
import itertools as it

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: list,
        pool_every: int,
        hidden_dims: list,
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions.
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        def output_shape(kernel_size, stride, pad):

            if type(kernel_size) is not tuple:
                kernel_size = (kernel_size, kernel_size)
            if type(stride) is not tuple:
                stride = (stride, stride)
            if type(pad) is not tuple:
                pad = (pad, pad)

            h = ((in_h + (2 * pad[0]) - kernel_size[0]) // stride[0]) + 1
            w = ((in_w + (2 * pad[1]) - kernel_size[1]) // stride[1]) + 1
            return h, w

        act = torch.nn.ReLU if self.activation_type=='relu' else torch.nn.LeakyReLU
        pool = torch.nn.MaxPool2d if self.pooling_type=='max' else torch.nn.AvgPool2d
        N = len(self.channels)
        P = self.pool_every

        for i in range(N):
            layer = nn.Conv2d(in_channels, self.channels[i], **self.conv_params)
            layers.append(layer)
            in_channels = self.channels[i]
            in_h, in_w = output_shape(layer.kernel_size, layer.stride, layer.padding)

            layers.append(act(**self.activation_params))

            if (i+1)%P==0:
                layer = pool(**self.pooling_params)
                layers.append(layer)
                in_h, in_w = output_shape(layer.kernel_size, layer.stride, layer.padding)

        layers.append(nn.Flatten())
        self.in_features = in_channels*in_h*in_w

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []
        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        act = torch.nn.ReLU if self.activation_type == 'relu' else torch.nn.LeakyReLU
        M = len(self.hidden_dims)
        input_features = self.in_features

        for i in range(M):
            layers.append(nn.Linear(input_features, self.hidden_dims[i]))
            layers.append(act(**self.activation_params))
            input_features = self.hidden_dims[i]

        layers.append(nn.Linear(input_features, self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_sizes: list,
        batchnorm=False,
        dropout=0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        layers = []
        act = torch.nn.ReLU if activation_type == 'relu' else torch.nn.LeakyReLU
        N = len(channels)
        out_channels = in_channels

        for i in range(N):
            channel = channels[i]
            kernel_size = kernel_sizes[i]
            layers.append(nn.Conv2d(out_channels, channel, kernel_size=kernel_size, padding=(kernel_size-1)//2))
            out_channels = channel
            if i<N-1:
                layers.append(nn.Dropout2d(dropout))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(channel))
                layers.append(act(**activation_params))
        self.main_path = nn.Sequential(*layers)
        self.shortcut_path = nn.Sequential(nn.Identity() if out_channels == in_channels else
                                           nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======

        def output_shape(kernel_size, stride, pad):

            if type(kernel_size) is not tuple:
                kernel_size = (kernel_size, kernel_size)
            if type(stride) is not tuple:
                stride = (stride, stride)
            if type(pad) is not tuple:
                pad = (pad, pad)

            h = ((in_h + (2 * pad[0]) - kernel_size[0]) // stride[0]) + 1
            w = ((in_w + (2 * pad[1]) - kernel_size[1]) // stride[1]) + 1
            return h, w

        pool = torch.nn.MaxPool2d if self.pooling_type == 'max' else torch.nn.AvgPool2d
        N = len(self.channels)
        P = self.pool_every

        for i in range(N//P):
            layer = ResidualBlock(
                in_channels=in_channels,
                channels=self.channels[i*P:(i+1)*P],
                kernel_sizes=[3]*P,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                activation_type=self.activation_type,
                activation_params=self.activation_params
            )
            layers.append(layer)

            in_channels = self.channels[(i+1)*P-1]

            layer = pool(**self.pooling_params)
            layers.append(layer)
            in_h, in_w = output_shape(layer.kernel_size, layer.stride, layer.padding)

        i = N%P
        if i>0:
            layer = ResidualBlock(
                in_channels=in_channels,
                channels=self.channels[-i:],
                kernel_sizes=[3] * i,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                activation_type=self.activation_type,
                activation_params=self.activation_params
            )
            layers.append(layer)

            in_channels = self.channels[-1]

        layers.append(nn.Flatten())
        self.in_features = in_channels * in_h * in_w

        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======

    # ========================
