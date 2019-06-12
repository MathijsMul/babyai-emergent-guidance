import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions import Categorical
import numpy as np
import itertools

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('GRU') != -1:
        for weight in [m.weight_ih_l0, m.weight_hh_l0]:
            weight.data.normal_(0, 1)
            weight.data *= 1 / torch.sqrt(weight.data.pow(2).sum(1, keepdim=True))
        for bias in [m.bias_ih_l0, m.bias_hh_l0]:
            bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))

class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out

class Corrector(nn.Module):
    """
    This is the Guide class

    One-to-many decoder that produces correction strings (guidance messages)
    """

    def __init__(self, image_dim=128, memory_dim=128, instr_dim=128, num_embeddings=3,
                 num_rnn_layers=1, vocabulary=None, max_tau=0.2, greedy=True, corr_length=2,
                 var_len=False, script=False,
                 obs_space=None):
        super().__init__()

        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.num_embeddings = num_embeddings
        self.num_rnn_layers = num_rnn_layers

        self.obs_space = obs_space

        self.var_len = var_len # variable correction lengths

        if vocabulary is not None:
            self.vocab = vocabulary # Vocabulary object, from obss_preprocessor / None
            self.vocab_idx2word = self.vocab.idx2word
            # Add SOS symbol to vocab/get idx
            self.sos_id = self.vocab['<S>']
        else:
            # if Corrector gets to use own vocabulary (standard)
            self.vocab_idx2word = {i: 'w' + str(i) for i in range(self.num_embeddings)}
            self.sos_id = 0
            if self.var_len:
                self.vocab_idx2word[self.num_embeddings] = '<eos>'
                self.eos_id = self.num_embeddings
                self.num_embeddings += 1
            self.vocab_word2idx = {self.vocab_idx2word[key] : key for key in self.vocab_idx2word}

        self.instr_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
        self.instr_rnn = nn.GRU(self.instr_dim, self.instr_dim, batch_first=True)

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            if ni < num_module-1:
                mod = ExpertControllerFiLM(
                    in_features=self.instr_dim,
                    out_features=128, in_channels=128, imm_channels=128)
            else:
                mod = ExpertControllerFiLM(
                    in_features=self.instr_dim, out_features=self.image_dim,
                    in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_Controler_' + str(ni), mod)

        self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

        self.word_embedding_corrector = nn.Embedding(num_embeddings=self.num_embeddings,
                                                         embedding_dim=self.instr_dim)

        self.decoder_rnn = nn.GRU(input_size=self.instr_dim,
                          hidden_size=self.memory_dim,
                          num_layers=self.num_rnn_layers,
                          batch_first=True)

        self.out = nn.Linear(self.memory_dim, self.num_embeddings)

        # learn tau(following https: // arxiv.org / pdf / 1701.08718.pdf) # Gumbel Softmax temperature
        self.tau_layer = nn.Sequential(nn.Linear(self.memory_dim, 1),
                                       nn.Softplus())
        self.max_tau = max_tau

        self.corr_length = corr_length # maximum length of correction message (if no variable length, always this length)
        self.greedy = greedy

        self.random_corrector = False

        if self.var_len:
            self.correction_loss = nn.CrossEntropyLoss()

        self.script = script

        self.apply(initialize_parameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward_film(self, instruction=None, observation=None, memory=None):
        instr_embedding = self._get_instr_embedding(instruction)
        x = torch.transpose(torch.transpose(observation, 1, 3), 2, 3)
        x = self.image_conv(x)
        for controler in self.controllers:
            x = controler(x, instr_embedding)
        film_output = F.relu(self.film_pool(x))

        film_output = film_output.reshape(x.shape[0], -1)

        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(film_output, hidden)
        memory_rnn_output = hidden[0]
        memory = torch.cat(hidden, dim=1)

        return(memory_rnn_output, memory)

    def forward(self, instruction=None, observation=None, memory=None, compute_message_probs=False, time=None):
        if not hasattr(self, 'random_corrector'):
            self.random_corrector = False
        if not hasattr(self, 'var_len'):
            self.var_len = False
        if not hasattr(self, 'script'):
            self.script = False

        if not self.script:
            memory_rnn_output, memory = self.forward_film(instruction=instruction,
                                                          observation=observation,
                                                          memory=memory)

            batch_size = instruction.size(0)
            correction_encodings = []

            entropy = 0.0

            lengths = np.array([self.corr_length] * batch_size)
            total_corr_loss = 0

            for i in range(self.corr_length):
                if i == 0:
                    # every message starts with a SOS token
                    decoder_input = torch.tensor([self.sos_id] * batch_size, dtype=torch.long, device=self.device)
                    decoder_input_embedded = self.word_embedding_corrector(decoder_input).unsqueeze(1)
                    decoder_hidden = memory_rnn_output.unsqueeze(0)

                if self.random_corrector:
                    # randomize corrections
                    device = torch.device("cuda" if decoder_input_embedded.is_cuda else "cpu")
                    decoder_input_embedded = torch.randn(decoder_input_embedded.size(), device=device)
                    decoder_hidden = torch.randn(decoder_hidden.size(), device=device)

                rnn_output, decoder_hidden = self.decoder_rnn(decoder_input_embedded, decoder_hidden)
                vocab_scores = self.out(rnn_output)
                vocab_probs = F.softmax(vocab_scores, dim=-1)

                entropy += Categorical(vocab_probs).entropy()

                tau = 1.0 / (self.tau_layer(decoder_hidden).squeeze(0) + self.max_tau)
                tau = tau.expand(-1, self.num_embeddings).unsqueeze(1)

                if self.training:
                    # Apply Gumbel SM
                    cat_distr = RelaxedOneHotCategorical(tau, vocab_probs)
                    corr_weights = cat_distr.rsample()
                    corr_weights_hard = torch.zeros_like(corr_weights, device=self.device)
                    corr_weights_hard.scatter_(-1, torch.argmax(corr_weights, dim=-1, keepdim=True), 1.0)

                    # detach() detaches the output from the computation graph, so no gradient will be backprop'ed along this variable
                    corr_weights = (corr_weights_hard - corr_weights).detach() + corr_weights

                else:
                    # greedy sample
                    corr_weights = torch.zeros_like(vocab_probs, device=self.device)
                    corr_weights.scatter_(-1, torch.argmax(vocab_probs, dim=-1, keepdim=True), 1.0)

                if self.var_len:
                    # consider sequence done when eos receives highest value
                    max_idx = torch.argmax(corr_weights, dim=-1)
                    eos_batches = max_idx.data.eq(self.eos_id)
                    if eos_batches.dim() > 0:
                        eos_batches = eos_batches.cpu().view(-1).numpy()
                        update_idx = ((lengths > i) & eos_batches) != 0
                        lengths[update_idx] = i

                    # compute correction error through pseudo-target: sequence of eos symbols to encourage short messages
                    pseudo_target = torch.tensor([self.eos_id for j in range(batch_size)], dtype=torch.long, device=self.device)
                    loss = self.correction_loss(corr_weights.squeeze(1), pseudo_target)
                    total_corr_loss += loss

                correction_encodings += [corr_weights]
                decoder_input_embedded = torch.matmul(corr_weights, self.word_embedding_corrector.weight)

            # one-hot vectors on forward, soft approximations on backward pass
            correction_encodings = torch.stack(correction_encodings, dim=1).squeeze(2)

            lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)

            result = {'correction_encodings' : correction_encodings,
                      'correction_messages' : self.decode_corrections(correction_encodings),
                      'correction_entropy' : torch.mean(entropy),
                      'corrector_memory' : memory,
                      'correction_lengths' : lengths,
                      'correction_loss' : total_corr_loss}

        else:
            # there is a script of pre-established guidance messages
            correction_messages = self.script[time]
            correction_encodings = self.encode_corrections(correction_messages)
            result = {'correction_encodings': correction_encodings,
                      'correction_messages': correction_messages}

        return (result)

    def _get_instr_embedding(self, instruction):
        _, hidden = self.instr_rnn(self.instr_embedding(instruction))
        x = hidden[-1]
        return(x)

    def _get_obs_embedding(self, observation):
        x = torch.transpose(torch.transpose(observation, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.squeeze(-1).squeeze(-1).unsqueeze(0)
        return(x)

    def decode_corrections(self, one_hot_encodings):
        max_prob = torch.topk(one_hot_encodings, k=1, dim=2)[1]
        batch_size = max_prob.size()[0]
        messages = [max_prob[message_idx] for message_idx in range(batch_size)]
        messages = [[self.vocab_idx2word.get(index.item(), '') for index in message] for message in messages]
        return(messages)

    def encode_corrections(self, correction_strings):
        if not hasattr(self, 'vocab_word2idx'):
            self.vocab_word2idx = {self.vocab_idx2word[key]: key for key in self.vocab_idx2word}
        print(correction_strings)
        indices = torch.tensor([[[self.vocab_word2idx[word]] for word in string] for string in correction_strings])
        batch_size = len(correction_strings)
        encodings = torch.zeros(batch_size, self.corr_length, self.num_embeddings).scatter_(2, indices, 1.0)
        return(encodings)

    def randomize(self):
        self.random_corrector = True

    def set_script(self, script):
        self.script = script

    def compute_all_messages(self):
        words = list(range(self.num_embeddings))
        all_messages = itertools.product(words, repeat=self.corr_length)
        all_encodings = torch.zeros((self.num_embeddings ** self.corr_length, self.corr_length, self.num_embeddings))

        #TODO: vectorize this from the start
        for idx, message in enumerate(all_messages):
            encoding = torch.zeros((self.corr_length, self.num_embeddings))

            for message_idx, word in enumerate(message):
                encoding[message_idx][word] = 1.0

            all_encodings[idx] = encoding

        print(all_encodings)

    def compute_message_probs(self, instruction, observation, memory, all_messages):
        # Use log probabilities

        memory_rnn_output, memory = self.forward_film(instruction=instruction,
                                                    observation=observation,
                                                    memory=memory)

        batch_size = instruction.size(0)

        def compute_prob_message(message):
            #TODO: make this more efficient. Now the entire chain of probabilities is computed every time...

            # message is tuple of form (0,2)
            prob = torch.zeros(batch_size, 1, device=self.device)

            for i in range(len(message)):
                if i == 0:
                    # every message starts with a SOS token
                    decoder_input = torch.tensor([self.sos_id] * batch_size, dtype=torch.long, device=self.device)
                    decoder_input_embedded = self.word_embedding_corrector(decoder_input).unsqueeze(1)
                    decoder_hidden = memory_rnn_output.unsqueeze(0)

                rnn_output, decoder_hidden = self.decoder_rnn(decoder_input_embedded, decoder_hidden)
                vocab_scores = self.out(rnn_output)
                vocab_probs = F.softmax(vocab_scores, dim=-1)

                step_prob = vocab_probs[:, :, message[i]]
                log_step_prob = torch.log(step_prob)
                prob += log_step_prob
                decoder_input_embedded = self.word_embedding_corrector(torch.tensor(message[i], device=self.device)).repeat(batch_size, 1).unsqueeze(1)

            return(prob.view(-1))

        message_probs = {message : compute_prob_message(message) for message in all_messages}
        return(message_probs)
