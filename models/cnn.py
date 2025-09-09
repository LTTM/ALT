import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.resnetcifar import ResNet50


class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNNMNIST_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return x, 0, y


class ProjectionHead(nn.Module):
    def __init__(self, num_ftrs, out_dim):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, h):
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x


class ModelFed(nn.Module):

    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFed, self).__init__()

        if base_model == "resnet50":
            basemodel = ResNet50()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        self.num_ftrs = num_ftrs
        self.project_features = ProjectionHead(num_ftrs, out_dim)
 
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def get_features(self, x):
        return self.features(x)

    def get_projected_features(self, x):
        return self.project_features(x)

    def forward(self, x):
        h = self.features(x)
        x = self.project_features(h)
        y = self.l3(x)
        return h, x, y


    def forward_fps(self, x1, x2):
        h1 = self.features(x1)
        x1 = self.project_features(h1)
        y1 = self.l3(x1)

        h2 = self.features(x2)
        x2 = self.project_features(h2)
        y2 = self.l3(x2)

        return y1, y2, x1.detach(), x2.detach()


class ModelFed_noheader(nn.Module):

    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFed_noheader, self).__init__()

        if base_model == "resnet50":
            basemodel = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        y = self.l3(h)
        return h, h, y

