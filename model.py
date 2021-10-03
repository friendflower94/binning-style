import torch
import torch.nn as nn
import torch.nn.functional as F

##### AdaIN #####
def adaptive_instance_normalization(content, style):
    assert(content.size()[:2]==style.size()[:2])
    size = content.size()

    mu_content, mu_style = torch.mean(content, axis=2), torch.mean(style, axis=2)
    sigma_content, sigma_style = torch.std(content, axis=2), torch.std(style, axis=2)

    normalized = (content - mu_content.expand(size)) / sigma_content.expand(size)

    return normalized * sigma_style.expand(size) + mu_style.expand(size)


##### gene discriminator #####
class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_layers = nn.Sequential(
            self.conv_relu(4, 128, 4, 1),
            self.conv_relu(128, 128, 3, 1),
            self.conv_relu(128, 256, 3, 1),
            self.conv_relu(256, 256, 3, 1),
            self.conv_relu(256, 512, 3, 1),
            self.conv_relu(512, 512, 3, 1),
            self.conv_relu(512, 1024, 3, 1),
            self.conv_relu(1024, 1024, 3, 1)
        )

        self.fc = nn.Linear(1024, output_dim)

        self.n_module = 8

    def forward(self, x):
        # module * 5
        x = self.conv_layers(x)

        # gap
        x = torch.mean(x, dim=2)

        # fc
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

    def conv_relu(self, input_dim, output_dim, kernel_size, padding):
        return nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

    def get_feature_map(self, x, layer):
        x = self.conv_layers[:layer](x)

        return x
    
    def get_gap(self,x):
        x = self.conv_layers(x)
        x = torch.mean(x, dim=2)

        return x    

    def get_style(self, x, layer):
        x = self.conv_layers[:layer](x)
        x = torch.squeeze(x)
        gram = torch.mm(x, torch.t(x))

        return gram

##### decoder #####
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_layers = nn.Sequential(
            self.conv_relu(64, 64, 3, 1),
            self.conv_relu(64, 32, 3, 1),
            self.conv_relu(32, 32, 3, 1),
            self.conv_relu(32, 4, 3, 1)
        )

    def conv_relu(self, input_dim, output_dim, kernel_size, padding):
        return nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(2)
        )

    def forward(self, x):
        return F.softmax(self.conv_layers(x), axis=1)
