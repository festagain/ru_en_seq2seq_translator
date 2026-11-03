import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.ru_ids = dataframe["ru_ids"].tolist()
        self.en_ids = dataframe["en_ids"].tolist()

    def __len__(self):
        return len(self.ru_ids)

    def __getitem__(self, index):
        source = torch.tensor(
            self.ru_ids[index],
            dtype=torch.long
        )

        target = torch.tensor(
            self.en_ids[index],
            dtype=torch.long
        )

        return source, target


class TranslationCollator:
    def __init__(self, ru_pad_idx, en_pad_idx):
        self.ru_pad_idx = ru_pad_idx
        self.en_pad_idx = en_pad_idx

    def __call__(self, batch):
        sources, targets = zip(*batch)

        source_lengths = torch.tensor(
            [len(source) for source in sources],
            dtype=torch.long
        )

        sources = pad_sequence(
            sources,
            batch_first=True,
            padding_value=self.ru_pad_idx
        )

        targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=self.en_pad_idx
        )

        return sources, source_lengths, targets