import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.hidden2out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        batch_size = features.shape[0]
        self.hidden = (torch.zeros((1, batch_size, self.hidden_size), device='cuda'), 
                       torch.zeros((1, batch_size, self.hidden_size), device='cuda'))
        embeddings = self.word_embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        out, self.hidden = self.lstm(embeddings, self.hidden)
        return self.hidden2out(out)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        batch_size = inputs.shape[0]
        hidden = (torch.zeros((1, batch_size, self.hidden_size), device='cuda'), 
                  torch.zeros((1, batch_size, self.hidden_size), device='cuda'))
        
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.hidden2out(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_indice = torch.max(outputs, dim=1)
            
            output.append(max_indice.cpu().numpy()[0].item())
            
            if (max_indice == 1):
                break
            
            inputs = self.word_embeddings(max_indice)
            inputs = inputs.unsqueeze(1)
            
        return output