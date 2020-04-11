import torch

from data.processed_dataset import ProcessedDataset
from models.cnn_seq2seq import CNNSeq2SeqModel


class RunnerCNNSeq2Seq():
    def __init__(self):
        super(RunnerCNNSeq2Seq, self).__init__()

        self.train_loader = torch.utils.data.DataLoader(
            ProcessedDataset(base_dir='../dataset/small/physionet_processed/',
                             temporal_len=10,
                             mode='train'),
            batch_size=2
        )

        self.model = CNNSeq2SeqModel()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.num_train_epochs = 20

        self.batch_log_interval = 20

    def train(self):

        lr = 5.0  # learning rate
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()  # Turn on the train mode
        for epoch_idx in range(self.num_train_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                X = batch[0].to(self.device)
                y = batch[1].to(self.device)

                optimizer.zero_grad()
                model_output = self.model(X)
                loss = criterion(model_output.reshape(-1, 5), torch.flatten(y))

                loss.backward()
                # TODO: gradient clipping for LSTM
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                if batch_idx % self.batch_log_interval == 0:
                    print('[Batch] Epoch: {:d} Batch: {:d} \t train_loss:{:5.2f}'.format(
                        epoch_idx, batch_idx, loss.item()))

                total_loss += loss.item()

            # Epoch end
            print('[Epoch End] Epoch: {:d} \t train_loss:{:5.2f}'.format(
                epoch_idx, total_loss))

            scheduler.step()


if __name__ == '__main__':
    runner = RunnerCNNSeq2Seq()

    runner.train()
