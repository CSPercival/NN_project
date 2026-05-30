import torch.nn as nn

class YOLOv1(nn.Module):

    def __init__(self, architecture, in_channels=3):
        super().__init__()

        self.architecture = architecture
        self.in_channels = in_channels
        self.model = self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):

        layers, _ = self.parse_architecture(
            self.architecture,
            self.in_channels
        )

        return nn.Sequential(*layers)

    def parse_architecture(self, architecture, in_channels):

        layers = []

        for layer in architecture:

            layer_type = layer[0]

            if layer_type == "Conv":

                _, kernel_size, filters, stride, padding = layer

                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False
                    )
                )

                layers.append(nn.BatchNorm2d(filters))
                layers.append(nn.LeakyReLU(0.1))

                in_channels = filters

            elif layer_type == "Maxpool":

                _, kernel_size, stride = layer

                layers.append(
                    nn.MaxPool2d(
                        kernel_size=kernel_size,
                        stride=stride
                    )
                )

            elif layer_type == "Repeat":

                _, n, subarchitecture = layer

                for _ in range(n):

                    new_layers, in_channels = self.parse_architecture(
                        subarchitecture,
                        in_channels
                    )

                    layers.extend(new_layers)

            elif layer_type == "Flatten":

                layers.append(nn.Flatten())

            elif layer_type == "FC":

                _, in_features, out_features = layer

                layers.append(
                    nn.Linear(
                        in_features,
                        out_features
                    )
                )

            elif layer_type == "Dropout":

                _, p = layer

                layers.append(nn.Dropout(p))

            elif layer_type == "LeakyReLU":

                _, x = layer

                layers.append(
                    nn.LeakyReLU(x)
                )

        return layers, in_channels