
from pytorch_lightning.utilities.model_summary import ModelSummary


def summary_models(model, return_flag=False):
    summary = ModelSummary(model)
    if return_flag:
        return summary
    else:
        print(summary)


if __name__ == "__main__":
    ##############################
    # region define test
    ##############################
    def test_summary_models():
        import torch
        import torch.nn as nn
        import pytorch_lightning as pl

        class MyLightningModel(pl.LightningModule):
            def __init__(self):
                super(MyLightningModel, self).__init__()
                self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.fc = nn.Linear(16 * 32 * 32, 10)

            def forward(self, x):
                x = self.conv(x)
                x = torch.relu(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        # Instantiate the model
        model = MyLightningModel()

        # Print the model summary
        summary = pl.utilities.model_summary.ModelSummary(model)
        print(summary)

    ##############################
    # region Run test
    ##############################
    test_summary_models()
