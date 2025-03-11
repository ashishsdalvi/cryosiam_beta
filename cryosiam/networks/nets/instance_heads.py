import torch.nn as nn


class InstanceHeads(nn.Module):
    """
    Build a InstanceHeads model.
    """

    def __init__(self, n_input_channels=1, spatial_dims=3, filters=(32, 64), kernel_size=3, padding=1):
        super().__init__()

        # create the encoder
        if spatial_dims == 2:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d

        self.foreground_head = nn.Sequential(conv(n_input_channels, filters[0], kernel_size, padding=padding),
                                             norm(filters[0]),
                                             nn.ReLU(inplace=True),
                                             conv(filters[0], filters[1], kernel_size, padding=padding),
                                             norm(filters[1]),
                                             nn.ReLU(inplace=True),
                                             conv(filters[1], 1, kernel_size, padding=padding))

        self.distance_head = nn.Sequential(conv(n_input_channels, filters[0], kernel_size, padding=padding),
                                           norm(filters[0]),
                                           nn.ReLU(inplace=True),
                                           conv(filters[0], filters[1], kernel_size, padding=padding),
                                           norm(filters[1]),
                                           nn.ReLU(inplace=True),
                                           conv(filters[1], 1, kernel_size, padding=padding))
        self.boundary_head = nn.Sequential(conv(n_input_channels, filters[0], kernel_size, padding=padding),
                                           norm(filters[0]),
                                           nn.ReLU(inplace=True),
                                           conv(filters[0], filters[1], kernel_size, padding=padding),
                                           norm(filters[1]),
                                           nn.ReLU(inplace=True),
                                           conv(filters[1], 1, kernel_size, padding=padding))

    def forward(self, x):
        # compute features for one view
        foreground = self.foreground_head(x)
        distances = self.distance_head(x)
        boundaries = self.boundary_head(x)

        return foreground, distances, boundaries
