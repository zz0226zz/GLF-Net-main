from Regression.libs import *

from Regression.data_load import PatientDataset, collate_fn


def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()


def main(args):

    gene_expression_file = args.gene_expression_file

    train_data = pd.read_csv(args.train_patient_id, index_col='patient_id')
    train_patient_ids = train_data.index.astype(str).tolist()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    base_model, num_features, preprocess = get_model(args.base_model_name)
    transform = preprocess

    dataset = PatientDataset(args.slides_dir, train_patient_ids, gene_expression_file, transform=transform)
    dataset_length = len(dataset)
    train_length = int(dataset_length * 0.8)  # Adjust train-validation split if needed
    valid_length = dataset_length - train_length

    # Initialize KFold for cross-validation
    n_splits = 5  # Number of folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=10)

    fold = 0
    max_r2_scores_per_fold = []

    for train_index, valid_index in kf.split(range(dataset_length)):
        fold += 1
        print(f"Fold {fold}/{n_splits}")

        # Split dataset using indices for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        valid_dataset = torch.utils.data.Subset(dataset, valid_index)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

        model = Feature(base_model).to(device)
        max_r2_score = float('-inf')  # Initialize to negative infinity

        for epoch in range(args.epochs):
            x_train = []
            y_train = []
            model.train()
            for images, gene_expression in tqdm(train_loader, desc=f"Fold {fold}/Epoch {epoch + 1}/Training"):
                images = tuple(batch.to(device) for batch in images)
                features = model(images)
                x_train.append(features.detach().cpu().numpy())
                y_train.append(gene_expression.detach().cpu().numpy())

            x_train = np.vstack(x_train)
            y_train = np.vstack(y_train)

            regressor = RandomForestRegressor(n_estimators=50, random_state=10)
            regressor.fit(x_train, y_train.ravel())

            y_pred_train = regressor.predict(x_train)
            train_loss = np.mean((y_pred_train - y_train.ravel()) ** 2)

            print(f'Fold {fold}, Epoch {epoch + 1}, train loss: {train_loss:.4f}')

            x_val = []
            y_val = []
            model.eval()

            with torch.no_grad():
                for images, gene_expression in tqdm(valid_loader, desc=f"Fold {fold}/Epoch {epoch + 1}/Validation"):
                    images = tuple(batch.to(device) for batch in images)
                    features = model(images).cpu().numpy()
                    x_val.append(features)
                    y_val.append(gene_expression.cpu().numpy())

            x_val = np.vstack(x_val)
            y_val = np.vstack(y_val)

            y_pred = regressor.predict(x_val)
            val_loss = np.mean((y_pred - y_val.ravel()) ** 2)

            r2_val = r2_score(y_val, y_pred)

            # Update max R² score for this fold
            if r2_val > max_r2_score:
                max_r2_score = r2_val

            print(f'Fold {fold}, Epoch {epoch + 1}, R² Score: {r2_val:.4f}, val loss: {val_loss:.4f}')

        # Store the maximum R² score for this fold
        max_r2_scores_per_fold.append(max_r2_score)

    # Compute and print average of maximum R² scores across all folds
    avg_max_r2_score = np.mean(max_r2_scores_per_fold)

    print(f"Cross-validation finished.")
    print(f'Average Maximum R² Score across all folds: {avg_max_r2_score:.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--epochs', type=int, default=20, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size')
    parser.add_argument('--base_model_name', type=str, default="GLF_Net", help='Base Model Name')
    parser.add_argument('--slides_dir', default="./bracregress", help="Directory for slides")
    parser.add_argument('--gene_expression_file', default="./brac.csv",
                        help="Path to the gene expression file in .csv")
    parser.add_argument('--train_patient_id', default="./brac_all.txt",
                        help="Path to the patient id file in .txt")
    args = parser.parse_args()

    main(args)