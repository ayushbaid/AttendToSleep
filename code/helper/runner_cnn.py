import os

import numpy as np
import torch


from data.processed_dataset import ProcessedDataset
from models.cnn import CNNModel


class RunnerCNN():
    def __init__(self):
        super(RunnerCNN, self).__init__()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.temporal_len = 1

        self.train_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/small/physionet_processed/train',
                             temporal_len=self.temporal_len,
                             mode='train'),
            batch_size=4,
            shuffle=True,
            pin_memory=self.cuda
        )

        self.test_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/small/physionet_processed/test',
                             temporal_len=self.temporal_len,
                             mode='train'),
            batch_size=4,
            shuffle=True,
            pin_memory=self.cuda
        )

        self.model = CNNModel().to(self.device)

        self.num_train_epochs = 20

        self.batch_log_interval = 500

        self.criterion = torch.nn.CrossEntropyLoss()

    def eval_model(self):
        self.model.eval()

        total_loss = 0
        total_correct_predictions = 0
        total_predictions = 0
        for batch_idx, batch in enumerate(self.test_loader):
            X = batch[0].to(self.device)
            y = torch.flatten(batch[1].to(self.device))

            model_output = self.model(X).reshape(-1, 5)
            loss = self.criterion(
                model_output, y)

            total_loss += loss.item()

            # compute the accuracy
            prediction = torch.argmax(model_output, dim=1)

            is_correct_prediction = torch.flatten(
                prediction == y
            ).detach().cpu().numpy().astype(int)

            total_correct_predictions += np.sum(
                is_correct_prediction, axis=None)
            total_predictions += is_correct_prediction.shape[0]

        self.model.train()
        return total_loss/self.test_loader.__len__(), float(total_correct_predictions)/total_predictions

    def train(self):

        lr = 1e-3  # learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        validation_loss, validation_accuracy = self.eval_model()

        print('[Start] val_loss:{:5.2f} \t  \t val_accuracy:{:1.2f}'.format(
            validation_loss, validation_accuracy))

        self.model.train()  # Turn on the train mode
        for epoch_idx in range(self.num_train_epochs):
            total_loss = 0
            total_correct_predictions = 0
            total_predictions = 0
            for batch_idx, batch in enumerate(self.train_loader):
                X = batch[0].to(self.device)
                y = torch.flatten(batch[1].to(self.device))

                optimizer.zero_grad()
                model_output = self.model(X).reshape(-1, 5)
                loss = self.criterion(
                    model_output, y)

                loss.backward()

                # compute the accuracy
                prediction = torch.argmax(model_output, dim=1)

                is_correct_prediction = torch.flatten(
                    prediction == y
                ).detach().cpu().numpy().astype(int)

                total_correct_predictions += np.sum(
                    is_correct_prediction, axis=None)
                total_predictions += is_correct_prediction.shape[0]

                optimizer.step()

                if batch_idx % self.batch_log_interval == 0:
                    print('[Batch] Epoch: {:d} Batch: {:d} \t train_loss:{:5.2f}'.format(
                        epoch_idx, batch_idx, loss.item()))

                total_loss += loss.item()

            # Epoch end
            train_loss = total_loss/self.train_loader.__len__()
            train_accuracy = float(total_correct_predictions)/total_predictions
            validation_loss, validation_accuracy = self.eval_model()

            print('[Epoch End] Epoch: {:d} \t train_loss:{:5.2f} \t val_loss:{:5.2f} \t train_accuracy:{:1.2f} \t val_accuracy:{:1.2f}'.format(
                epoch_idx, train_loss, validation_loss, train_accuracy, validation_accuracy))

            scheduler.step()

    def save_model(self):
        torch.save({
            'model_cnnLeft': self.model.cnn_left.state_dict(),
            'model_cnnRight': self.model.cnn_right.state_dict(),
        }, os.path.join('../models/', 'cnn_checkpoint.pt'))


if __name__ == '__main__':
    runner = RunnerCNN()

    runner.train()

    runner.save_model()
