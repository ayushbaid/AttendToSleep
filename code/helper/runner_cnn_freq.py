import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import sklearn.metrics as metrics


from data.processed_dataset import ProcessedDataset
from models.cnn_freq import CNNFreqModel


class RunnerCNNFreq():
    def __init__(self):
        super(RunnerCNNFreq, self).__init__()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.temporal_len = 1

        self.train_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/all/processed/train',
                             temporal_len=self.temporal_len,
                             mode='train'),
            batch_size=200,
            shuffle=True,
            pin_memory=self.cuda
        )

        self.val_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/all/processed/val',
                             temporal_len=self.temporal_len,
                             mode='train'),
            batch_size=200,
            shuffle=True,
            pin_memory=self.cuda
        )

        self.test_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/all/processed/test',
                             temporal_len=self.temporal_len,
                             mode='test'),
            batch_size=1,
            shuffle=True,
            pin_memory=self.cuda
        )

        self.model = CNNFreqModel().to(self.device)

        self.num_train_epochs = 10

        self.batch_log_interval = 500

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # class-weghting for loss
        class_entries = np.array(
            [12241.0, 3786.0, 13320.0, 3658.0, 4609.0],
            dtype=np.float64
        )

        loss_weights = 1.0/class_entries
        loss_weights = loss_weights/np.sum(loss_weights)

        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(loss_weights).to(self.device)
        )

    def eval_model(self):
        '''
        Evaluate on the validation set
        '''
        self.model.eval()

        total_loss = 0
        total_correct_predictions = 0
        total_predictions = 0
        for batch in self.val_loader:
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
        return total_loss/self.val_loader.__len__(), float(total_correct_predictions)/total_predictions

    def test_model(self):
        '''
        Evaluate on the test set
        '''
        self.model.eval()

        target_list = []
        prediction_list = []

        for batch in self.test_loader:
            X = batch[0].to(self.device)
            y = torch.flatten(batch[1].to(self.device))

            model_output = self.model(X).reshape(-1, 5)

            # compute the accuracy
            prediction_list.append(torch.flatten(torch.argmax(
                model_output, dim=1
            )).detach().cpu().numpy().astype(int))
            target_list.append(
                y.detach().cpu().numpy().astype(int)
            )

        self.model.train()

        # make a single array
        prediction_array = np.concatenate(prediction_list)
        target_array = np.concatenate(target_list)

        prediction_list = None
        target_list = None

        f1_score = metrics.f1_score(
            target_array, prediction_array, average='macro')
        accuracy = metrics.accuracy_score(target_array, prediction_array)

        confusion_mat = metrics.confusion_matrix(
            target_array, prediction_array)

        return accuracy, f1_score, confusion_mat

    def train(self):

        lr = 1e-3  # learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        validation_loss, validation_accuracy = self.eval_model()

        print('[Start] val_loss:{:5.2f} \t  \t val_accuracy:{:1.2f}'.format(
            validation_loss, validation_accuracy))

        self.train_loss_history = []
        self.val_loss_history = [validation_loss]
        self.train_acc_history = []
        self.val_acc_history = [validation_accuracy]

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

                # save the 0th batch of 0th epoch
                if epoch_idx == 0 and batch_idx == 0:
                    self.train_loss_history.append(loss.item())
                    self.train_acc_history.append(np.sum(
                        is_correct_prediction, axis=None)/is_correct_prediction.shape[0]
                    )

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

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(validation_loss)
            self.train_acc_history.append(train_accuracy)
            self.val_acc_history.append(validation_accuracy)

            self.save_loss_curves()

            scheduler.step()

    def save_model(self):
        torch.save({
            'cnn': self.model.cnn.state_dict(),
        }, os.path.join('../models/', 'cnn_freq_checkpoint.pt'))

    def save_loss_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(self.train_loss_history, label='train')
        ax1.plot(self.val_loss_history, label='val')

        ax1.set_title('Loss curve')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()

        ax2.plot(self.train_acc_history, label='train')
        ax2.plot(self.val_acc_history, label='val')

        ax2.set_title('Accuracy curve')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.tight_layout()

        plt.savefig('../plots/cnn_freq_learning_curves.png')

        plt.close()


if __name__ == '__main__':
    runner = RunnerCNNFreq()

    runner.train()

    runner.save_model()

    runner.save_loss_curves()

    test_f1, test_accuracy, test_confusion = runner.test_model()

    print('==== Test set evaluation ====')
    print(
        'Accuracy: {:0.2f} \t F1-score: {:0.2f}'.format(test_accuracy, test_f1)
    )
    print('Confusion Matrix: ')
    print(test_confusion)
