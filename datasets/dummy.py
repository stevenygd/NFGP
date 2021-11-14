from torch.utils import data
from torch.utils.data import Dataset


class Dummy(Dataset):

    def __init__(self, cfgdata):
        self.length = int(cfgdata.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {}


def get_data_loaders(cfg, args):
    tr_dataset = Dummy(cfg.train)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=1,
        shuffle=False, num_workers=0, drop_last=False)

    te_dataset = Dummy(cfg.val)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=1,
        shuffle=False, num_workers=0, drop_last=False)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders