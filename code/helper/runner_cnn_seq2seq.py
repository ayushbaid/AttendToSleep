import numpy as np
import torch


from data.processed_dataset import ProcessedDataset
from models.cnn_seq2seq import CNNSeq2SeqModel


class RunnerCNNSeq2Seq():
    def __init__(self):
        super(RunnerCNNSeq2Seq, self).__init__()

        self.temporal_len = 10

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.train_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/all/processed/train',
                             temporal_len=self.temporal_len,
                             mode='train'),
            batch_size=4,
            shuffle=True,
            pin_memory=self.cuda
        )

        self.test_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/all/processed/test',
                             temporal_len=self.temporal_len,
                             mode='train'),
            batch_size=4,
            shuffle=True,
            pin_memory=self.cuda
        )

        self.model = CNNSeq2SeqModel(
            num_temporal=self.temporal_len).to(self.device)

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

        lr = 0.1  # learning rate
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        validation_loss, validation_accuracy = self.eval_model()

        print('[Start] val_loss:{:5.2f} \t  \t val_accuracy:{:1.2f}'.format(validation_loss, validation_accuracy))

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

                # TODO: gradient clipping for LSTM
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
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


if __name__ == '__main__':
    runner = RunnerCNNSeq2Seq()

    runner.train()
