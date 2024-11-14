import os
import torch
import random
from torch import nn
import torch.utils.data
from tqdm import tqdm

flog = None


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, id_list, args):
        super(Dataset, self).__init__()
        self.device = args.device
        self.choose = args.choose
        self.node = args.node
        self.input_size = args.input_size
        self.data = []
        for id, gender in tqdm(id_list, file=flog):
            fp = open(os.path.join(args.data_dir, "%d.txt" % id), "r")
            s = fp.readlines()
            fp.close()
            for i in range(len(s)):
                s[i] = [float(item) for item in s[i].split(' ')]
            self.data.append((torch.tensor(s, device=args.device).reshape(-1, args.node, args.node), torch.tensor(gender, device=args.device).reshape(-1)))

    def __getitem__(self, index):
        if (self.choose):
            return (self.data[index][0][random.randrange(self.input_size), :].reshape(-1, self.node, self.node), self.data[index][1])
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.cnn = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Conv2d(1 if args.choose else args.input_size, args.channel, kernel_size=args.node),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(0),
            nn.Dropout(args.dropout),
            nn.Linear(args.channel, args.channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(args.channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, mat):
        return self.cnn(mat).reshape(1)
