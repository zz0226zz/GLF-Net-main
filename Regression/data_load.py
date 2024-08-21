from Regression.libs import *

class PatientDataset(Dataset):

    def __init__(self, slides_dir, patient_ids, gene_expression_file, transform=None):
        self.slides_dir = slides_dir
        self.gene_expression_data = pd.read_csv(gene_expression_file, index_col='PATIENT_ID')

        self.patient_ids = patient_ids

        self.model_transform = transform

        self.preprocess = transforms.Compose([

            self.model_transform,
        ])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        img_paths = sorted(glob.glob(os.path.join(self.slides_dir, patient_id, '*.png')))
        images = []
        for img_path in img_paths:
            img = Image.open(img_path)
            img = self.preprocess(img)
            images.append(img)

        images = torch.stack(images)
        gene_expression = self.gene_expression_data.loc[patient_id]
        gene_expression = torch.tensor(gene_expression.values, dtype=torch.float32)
        return images, gene_expression


def collate_fn(batch):
    images, patient_id = zip(*batch)
    clipped_images = []
    for img in images:
        if len(img) > 100:
            img = img[:100]
        clipped_images.append(img)
    gene_expression = torch.stack([expr for _, expr in batch], dim=0)
    return clipped_images, gene_expression
