from Regression.libs import *

class Feature(nn.Module):

    def __init__(self, base_model):
        super(Feature, self).__init__()
        self.base_model = base_model

    def forward(self, x):

        aggregated_features_list = []

        for patient_images in x:
            patient_features = self.base_model(patient_images)
            aggregated_feature = patient_features.mean(dim=0)
            aggregated_features_list.append(aggregated_feature)
            aggregated_features = torch.stack(aggregated_features_list, dim=0)

        return aggregated_features