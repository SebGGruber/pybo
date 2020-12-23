import numpy as np
import torch
import torchvision.models.resnet as resnet
from tqdm import tqdm


class ResNet_channels(resnet.ResNet):

    def __init__(self, in_channels, *args, **kwargs):
        super(ResNet_channels, self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)

        return logits


def _resnet(arch, block, layers, pretrained, progress, in_channels, **kwargs):
    model = ResNet_channels(in_channels, block, layers, **kwargs)
    return model


def resnet10(
    pretrained=False, progress=True, in_channels=1, **kwargs
):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Modified from https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet10",
        resnet.BasicBlock,
        [1, 1, 1, 1],
        pretrained,
        progress,
        in_channels,
        **kwargs
    )


def train(model, optimizer, dataset, epochs, device):

    # progress bar displayed in the console output
    with tqdm(
        total=epochs, desc="Pytorch model training", disable=False, leave=False
    ) as progress_bar:

        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        val_losses = []

        for epoch in range(epochs):

            # model fitting
            for (inputs, labels) in dataset["train"]:

                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

                # update model on mini-batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # calculate validation loss
            with torch.no_grad():

                for (inputs, labels) in dataset["val"]:
                    
                    inputs = inputs.to(device=device)
                    labels = labels.to(device=device)

                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels).item()
                    val_losses.append(val_loss)

            postfix = {"val_loss": val_loss}
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)

    # return the best validation loss of all epochs
    return np.min(val_losses)

