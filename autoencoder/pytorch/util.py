import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pickle
import numpy as np

from torch.autograd import Variable

def obtain_dataloader(batch_size):
    """
    obtain the data loader
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = dset.MNIST(root='./data', train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

class Autoencoder(nn.Module):
    """
    Autoencoder
    """

    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, activation):
        """
        specify an autoencoder
        """
        assert input_dim == output_dim, 'The input and output dimension should be the same'
        self.encoder_weight = torch.randn([input_dim, hidden_dim]) * 0.02
        self.decoder_weight = torch.randn([hidden_dim, output_dim]) * 0.02
        self.batch_size = batch_size

        if activation.lower() in ['relu']:
            self.activation=lambda x: torch.clamp(x, min = 0.)
            self.dactivation=lambda x: (torch.sign(x) + 1) / 2
        elif activation.lower() in ['tanh']:
            self.activation=lambda x: (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
            self.dactivation=lambda x: 4 / (torch.exp(x) + torch.exp(-x)) ** 2
        elif activation.lower() in ['identity']:
            self.activation=lambda x: x
            self.dactivation=lambda x: torch.ones_like(x)
        elif activation.lower() in ['sigmoid', 'sigd']:
            self.activation=lambda x: 1. / (1. + torch.exp(-x))
            self.dactivation=lambda x: torch.exp(x) / (1 + torch.exp(x)) ** 2
        elif activation.lower() in ['negative']:
            self.activation=lambda x: -x
            self.dactivation=lambda x: -torch.ones_like(x)
        else:
            raise ValueError('unrecognized activation function')

    def train(self, data_batch, step_size):
        """
        training a model

        :param data_batch: of shape [batch_size, input_dim]
        :param step_size: float, step size
        """
       
        projection = torch.matmul(data_batch, self.encoder_weight)
        
        
        encode = self.activation(projection)
        
        dencode = self.dactivation(projection)
        
        decode = torch.matmul(encode, self.decoder_weight)  # of shape [batch_size, output_dim]
        
        
        error = decode - data_batch
        loss = torch.mean(torch.sum(error ** 2, dim = 1)) / 2

        # TODO calculate the gradient and update the weight
        # NO autograd is allowed
        b  = data_batch.shape[0]
        
        (d,k) = self.encoder_weight.shape
        
        
        grad_decode_loss = torch.zeros_like(self.decoder_weight)
        grad_encode_loss = torch.zeros_like(self.encoder_weight)

        for i in range(b):
            grad_decode_loss = grad_decode_loss - torch.mm(encode[i].view(k,1),(data_batch[i] - decode[i]).view(1,d))
            grad_encode_loss = grad_encode_loss - torch.mm(data_batch[i].view(d,1),torch.mm((data_batch[i] - decode[i]).view(1,d), 
                                               self.decoder_weight.view(d,k)) * dencode[i].view(1,k))
        self.encoder_weight = self.encoder_weight - step_size * grad_encode_loss / b
        self.decoder_weight = self.decoder_weight - step_size * grad_decode_loss / b
            
        return float(loss)

    def test(self, data_batch):
        """
        test and calculate the reconstruction loss
        """
        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)
        decode = torch.matmul(encode, self.decoder_weight)

        error = decode - data_batch
        loss = torch.mean(torch.sum(error ** 2, dim = 1)) / 2

        return loss

    def compress(self, data_batch):
        """
        compress a data batch
        """

        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)

        return np.array(encode)

    def reconstruct(self, data_batch):
        """
        reconstruct the image
        """
        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)
        decode = torch.matmul(encode, self.decoder_weight)

        return np.array(decode)

    def save_model(self, file2dump):
        """
        save the model
        """
        pickle.dump(
                [np.array(self.encoder_weight), np.array(self.decoder_weight)],
                open(file2dump, 'wb'))

    def load_model(self, file2load):
        """
        load the model
        """
        encoder_weight, decoder_weight = pickle.load(open(file2load, 'rb'))
        self.encoder_weight = torch.FloatTensor(encoder_weight)
        self.decoder_weight = torch.FloatTensor(decoder_weight)
