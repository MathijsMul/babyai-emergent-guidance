import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import itertools
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads
from babyai.corrector import Corrector

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('GRU') != -1:
        # this part is new!
        for weight in [m.weight_ih_l0, m.weight_hh_l0]:
            weight.data.normal_(0, 1)
            weight.data *= 1 / torch.sqrt(weight.data.pow(2).sum(1, keepdim=True))
        for bias in [m.bias_ih_l0, m.bias_hh_l0]:
            bias.data.fill_(0)

    elif classname.find('Embedding') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871

class AgentControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=imm_channels, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        return self.conv(x) * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)


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


class ImageBOWEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, reduce_fn=torch.mean):
        super(ImageBOWEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.reduce_fn = reduce_fn
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.reduce_fn(embeddings, dim=1)
        embeddings = torch.transpose(torch.transpose(embeddings, 1, 3), 2, 3)
        return embeddings

class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None,
                 vocabulary=None,
                 learner=False,
                 corrector=False, corr_length=3, corr_own_vocab=False, corr_vocab_size=0,
                 pretrained_corrector=False, use_critic=False, dropout=0.5, corrector_frozen=False, random_corrector=False,
                 var_corr_len=False, weigh_corrections=False,
                 return_internal_repr=False):
        super().__init__()

        self.vocab = vocabulary # Vocabulary object, from obss_preprocessor

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.use_learner = learner
        self.use_corrector = corrector
        self.corr_own_vocab = corr_own_vocab
        self.pretrained_corrector = pretrained_corrector

        self.obs_space = obs_space

        self.policy_input_size = 0
        self.use_critic = use_critic

        if self.use_learner:
            if arch == "cnn1":
                self.image_conv = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
                    nn.ReLU()
                )
            elif arch == "cnn2":
                self.image_conv = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True),
                    nn.Conv2d(in_channels=16, out_channels=image_dim, kernel_size=(3, 3)),
                    nn.ReLU()
                )
            elif arch == "filmcnn":
                if not self.use_instr:
                    raise ValueError("FiLM architecture can be used when instructions are enabled")

                self.image_conv_1 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=2)
                )
                self.image_conv_2 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(2, 2)),
                    nn.ReLU()
                )
            elif arch.startswith("expert_filmcnn"):
                if not self.use_instr:
                    raise ValueError("FiLM architecture can be used when instructions are enabled")

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
            elif arch == 'embcnn1':
                self.image_conv = nn.Sequential(
                    ImageBOWEmbedding(obs_space["image"], embedding_dim=16, padding_idx=0, reduce_fn=torch.mean),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(3, 3)),
                    nn.ReLU()
                )
            else:
                raise ValueError("Incorrect architecture name: {}".format(arch))

            # Define instruction embedding
            if self.use_instr:
                if self.lang_model in ['gru', 'conv', 'bigru', 'attgru']:
                    self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                    if self.lang_model in ['gru', 'bigru', 'attgru']:
                        gru_dim = self.instr_dim
                        if self.lang_model in ['bigru', 'attgru']:
                            gru_dim //= 2
                        self.instr_rnn = nn.GRU(
                            self.instr_dim, gru_dim, batch_first=True,
                            bidirectional=(self.lang_model in ['bigru', 'attgru']))
                        self.final_instr_dim = self.instr_dim
                    else:
                        kernel_dim = 64
                        kernel_sizes = [3, 4]
                        self.instr_convs = nn.ModuleList([
                            nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                        self.final_instr_dim = kernel_dim * len(kernel_sizes)

                elif self.lang_model == 'bow':
                    hidden_units = [obs_space["instr"], self.instr_dim, self.instr_dim]
                    layers = []
                    for n_in, n_out in zip(hidden_units, hidden_units[1:]):
                        layers.append(nn.Linear(n_in, n_out))
                        layers.append(nn.ReLU())
                    self.instr_bow = nn.Sequential(*layers)
                    self.final_instr_dim = instr_dim

                if self.lang_model == 'attgru':
                    self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            # Define memory
            if self.use_memory:
                self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
                self.policy_input_size += self.memory_dim

            # Resize image embedding
            self.embedding_size = self.semi_memory_size
            if self.use_instr and arch != "filmcnn" and not arch.startswith("expert_filmcnn"):
                self.embedding_size += self.final_instr_dim

            if arch == "filmcnn":
                self.controller_1 = AgentControllerFiLM(
                    in_features=self.final_instr_dim, out_features=64,
                    in_channels=3, imm_channels=16)
                self.controller_2 = AgentControllerFiLM(
                    in_features=self.final_instr_dim,
                    out_features=64, in_channels=32, imm_channels=32)

            if arch.startswith("expert_filmcnn"):
                if arch == "expert_filmcnn":
                    num_module = 2
                else:
                    num_module = int(arch[(arch.rfind('_') + 1):])
                self.controllers = []
                for ni in range(num_module):
                    if ni < num_module-1:
                        mod = ExpertControllerFiLM(
                            in_features=self.final_instr_dim,
                            out_features=128, in_channels=128, imm_channels=128)
                    else:
                        mod = ExpertControllerFiLM(
                            in_features=self.final_instr_dim, out_features=self.image_dim,
                            in_channels=128, imm_channels=128)
                    self.controllers.append(mod)
                    self.add_module('FiLM_Controler_' + str(ni), mod)

        # Initialize parameters correctly.
        # Put this here, because otherwise pretrained corrector's linear weights are overwritten
        self.apply(initialize_parameters)

        if self.use_corrector:
            self.corr_vocab_size = corr_vocab_size
            self.corr_length = corr_length
            self.var_corr_len = var_corr_len

            if not self.pretrained_corrector:
                if self.corr_own_vocab:
                    num_corr_embeddings = corr_vocab_size
                    vocabulary_corr = None
                else:
                    num_corr_embeddings = self.obs_space['instr']
                    vocabulary_corr = self.vocab

                self.corrector = Corrector(image_dim=self.image_dim,
                                               memory_dim=self.memory_dim,
                                               instr_dim=self.instr_dim,
                                               num_embeddings=num_corr_embeddings,
                                               num_rnn_layers=1,
                                               vocabulary=vocabulary_corr,
                                               corr_length=self.corr_length,
                                               obs_space=self.obs_space,
                                               var_len=self.var_corr_len)

            else:
                self.load_pretrained_corrector(self.pretrained_corrector, corrector_frozen=corrector_frozen)
                corr_vocab_size = self.corrector.word_embedding_corrector.num_embeddings

            if self.corr_own_vocab:
                if self.var_corr_len:
                    num_corr_embeddings = corr_vocab_size + 1
                else:
                    num_corr_embeddings = corr_vocab_size

                self.word_embedding_corrections = nn.Embedding(num_corr_embeddings, self.instr_dim)
                corr_rnn_hidden = 512 # currently constant
                self.corr_rnn = nn.GRU(input_size=self.instr_dim,
                                       hidden_size=corr_rnn_hidden,
                                       batch_first=True)
            else:
                if self.use_learner:
                    self.word_embedding_corrections = self.word_embedding
                    self.corr_rnn = self.instr_rnn
                else:
                    self.word_embedding_corrections = self.corrector.instr_embedding
                    self.corr_rnn = self.corrector.instr_rnn

            self.policy_input_size += self.corr_rnn.hidden_size

            self.corr_dropout = nn.Dropout(p=dropout)

            if random_corrector:
                self.corrector.randomize()

            self.weigh_corrections = weigh_corrections
            if self.weigh_corrections:
                # parameter to determine weight of corrections from previous entropy
                self.entropy_weight = nn.Linear(1, 1) #nn.Parameter(torch.randn(1))

        if self.use_critic:
            # Define critic's model (used in PPO, not in IL)
            self.critic = nn.Sequential(
                nn.Linear(self.policy_input_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.policy_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

        self.return_internal_repr = return_internal_repr

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def load_pretrained_corrector(self, pretrained_corrector, corrector_frozen=False):
        # Load Corrector from pretrained model
        print('Loading pretrained corrector from:')
        print(pretrained_corrector)

        if not torch.cuda.is_available():
            self.corrector = torch.load('./models/' + pretrained_corrector + '/model.pt',
                                        map_location='cpu').corrector
            self.corrector.device = "cpu"
        else:
            self.corrector = torch.load('./models/' + pretrained_corrector + '/model.pt').corrector
            self.corrector.device = "cuda"

        if corrector_frozen:
            for param in self.corrector.parameters():
                param.requires_grad = False

        # in case a higher level requires more word embeddings, just add 15 extra rows to matrix (should be enough for sure)
        extra_emb = torch.zeros((15, self.corrector.instr_dim), requires_grad=True)

        # initialize extra embeddings properly
        extra_emb.data.normal_(0, 1)
        extra_emb.data *= 1 / torch.sqrt(extra_emb.data.pow(2).sum(1, keepdim=True))

        new_emb = nn.Parameter(torch.cat((self.corrector.instr_embedding.weight.data, extra_emb)))
        self.corrector.instr_embedding.weight = new_emb
        self.corrector.instr_embedding.num_embeddings += 15

    def forward_film(self, instruction=None, observation=None, memory=None, instr_embedding=None):
        if self.use_instr and instr_embedding is None:

            instr_embedding = self._get_instr_embedding(instruction)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M

            mask = (instruction != 0).float()

            max_length = instr_embedding.shape[1]
            if mask.shape[1] > max_length:
                mask = mask[:, :max_length]

            instr_embedding = instr_embedding[:, :mask.shape[1]]
            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask

            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(observation, 1, 3), 2, 3)

        if self.arch == "filmcnn":
            x = self.controller_1(x, instr_embedding)
            x = self.image_conv_1(x)
            x = self.controller_2(x, instr_embedding)
            x = self.image_conv_2(x)
        elif self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and self.arch != "filmcnn" and not self.arch.startswith("expert_filmcnn"):
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        return(embedding, memory)

    def forward(self, obs, memory=None, instr_embedding=None, memory_corrector=None, correction_encodings=None, previous_entropy=None, time=None):
        # to decode instruction:
        # instr_list = [list(message) for message in list(obs.instr)]
        # decoded_instr = [[self.vocab.idx2word.get(index.item(), '') for index in sentence] for sentence in instr_list]
        # print(decoded_instr)

        value, corrections, extra_predictions, correction_result, internal_repr, correction_messages, correction_loss, corr_weight = None, None, None, None, None, None, None, None
        policy_input = []

        # remove this at some point:
        if not hasattr(self, 'return_internal_repr'):
            self.return_internal_repr = False
        if not hasattr(self, 'var_corr_len'):
            self.var_corr_len = False
        if not hasattr(self, 'use_learner'):
            self.use_learner = True
        if not hasattr(self, 'use_critic'):
            self.use_critic = False
        if not hasattr(self, 'weigh_corrections'):
            self.weigh_corrections = False

        if self.use_learner:
            embedding, memory = self.forward_film(instruction=obs.instr,
                                          observation=obs.image,
                                          memory=memory,
                                          instr_embedding=instr_embedding)

            policy_input += [embedding]

        if self.use_corrector:
            if correction_encodings is None:
                correction_result = self.corrector(instruction=obs.instr,
                                                   observation=obs.image,
                                                   memory=memory_corrector,
                                                   time=time)

                correction_encodings = correction_result['correction_encodings']
                correction_messages = correction_result['correction_messages']

                if not self.corrector.script:
                    corr_lengths = correction_result['correction_lengths']
                    correction_loss = correction_result['correction_loss']
                    memory_corrector = correction_result['corrector_memory']

            correction_embeddings = torch.matmul(correction_encodings, self.word_embedding_corrections.weight)

            if self.var_corr_len:
                # zero out zero-length corrections, and give them dummy length 1 (breaks computation graph though)
                batch_size = obs.instr.size(0)
                mask = corr_lengths.data.ne(0)
                mask = mask.view(-1, 1, 1).expand(batch_size, self.corr_length, self.instr_dim).float()
                correction_embeddings *= mask
                zero_length_inds = corr_lengths.data.eq(0).long()
                corr_lengths += zero_length_inds

                seq_lengths, perm_idx = corr_lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if correction_embeddings.is_cuda:
                    iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                correction_embeddings = correction_embeddings[perm_idx]
                correction_embeddings = pack_padded_sequence(correction_embeddings, seq_lengths,
                                                  batch_first=True)  # seq_lengths.data.cpu().numpy()

            _, hidden = self.corr_rnn(correction_embeddings)
            corr_embedding = hidden[-1]

            if self.var_corr_len:
                # unpermute embeddings
                corr_embedding = corr_embedding[iperm_idx]

            if not self.pretrained_corrector:
                # dropout layer for corr_embedding => very important to make corrector's input to policy more varied.
                corr_embedding = self.corr_dropout(corr_embedding)

            if self.weigh_corrections:
                corr_weight = self.entropy_weight(previous_entropy)
                corr_weight = nn.Sigmoid()(corr_weight)
                corr_embedding *= corr_weight

            policy_input += [corr_embedding]

        policy_input = torch.cat(policy_input, dim=1)

        if self.return_internal_repr:
            # return desired internal representations for analysis/visualization
            all_policy_input = torch.split(policy_input, split_size_or_sections=1)

            policy_inputs = [input[0].tolist() for input in all_policy_input]
            policy_layers = [torch.sum(torch.mul(input, self.actor[0].weight), dim=0).tolist() for input in all_policy_input]

            internal_repr = {'policy_inputs' : policy_inputs,
                             'policy_layers' : policy_layers}

        x = self.actor(policy_input)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        if self.use_critic:
            x = self.critic(policy_input)
            value = x.squeeze(1)

        # argmax_action = dist.probs.max(1, keepdim=True)[1]
        # print(argmax_action)

        return {'dist': dist,
                'value': value,
                'memory': memory,
                'extra_predictions': extra_predictions,
                'corrections': correction_messages,
                'memory_corrector' : memory_corrector,
                'correction_encodings': correction_encodings,
                'internal_repr': internal_repr,
                'correction_loss': correction_loss,
                'correction_weight': corr_weight}

    def compute_cic(self, obs, memory=None, instr_embedding=None, memory_corrector=None, previous_entropy=None):
        # compute causal influence of communication (CIC) (using log probs)

        device = torch.device("cuda" if memory.is_cuda else "cpu")

        batch_size = obs.instr.size(0)
        all_messages = list(itertools.product(list(range(self.corrector.num_embeddings)), repeat=self.corrector.corr_length))

        log_probs_messages = self.corrector.compute_message_probs(instruction=obs.instr,
                                                             observation=obs.image,
                                                             memory=memory_corrector,
                                                             all_messages=all_messages)

        def compute_action_probs(message):
            message_indices = [[message[i]] for i in range(len(message))]
            message_encoding = torch.zeros(self.corrector.corr_length, self.corrector.num_embeddings, device=device).scatter_(1, torch.tensor(message_indices, device=device), 1.0).unsqueeze(0).repeat(batch_size, 1, 1)
            model_result = self.forward(obs=obs,memory=memory,instr_embedding=instr_embedding,memory_corrector=memory_corrector, correction_encodings=message_encoding, previous_entropy=previous_entropy)
            probs = model_result['dist'].probs
            log_probs = torch.log(probs)
            return(log_probs)

        log_probs_action_given_message = {message : compute_action_probs(message) for message in all_messages}
        log_probs_joint_action_message = {message : {action : log_probs_messages[message] + log_probs_action_given_message[message][:,action] for action in range(7)} for message in all_messages}
        log_probs_actions = {action : torch.log(torch.sum(torch.stack([torch.exp(log_probs_joint_action_message[message][action]) for message in all_messages], dim=1), dim=1)) for action in range(7)}

        # for idx in range(batch_size):
            # sanity check: do these sum up to 1?
            # print(torch.sum(torch.stack([probs_actions[action][idx] for action in range(7)])))

        cic = 0

        for idx, message in enumerate(all_messages):
            # compute mutual information
            for action in range(7):
                log_p_am = log_probs_joint_action_message[message][action]

                # compute mutual information
                cic += torch.exp(log_p_am) * (log_p_am - log_probs_actions[action] - log_probs_messages[message])

        return(cic)

    def _get_instr_embedding(self, instr):
        if self.lang_model == 'gru':
            _, hidden = self.instr_rnn(self.word_embedding(instr))
            return hidden[-1]

        elif self.lang_model in ['bigru', 'attgru']:
            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            if outputs.shape[1] < masks.shape[1]:
                masks = masks[:, :(outputs.shape[1]-masks.shape[1])]
                # the packing truncated the original length
                # so we need to change mask to fit it

            return outputs if self.lang_model == 'attgru' else final_states

        elif self.lang_model == 'conv':
            inputs = self.word_embedding(instr).unsqueeze(1)  # (B,1,T,D)
            inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.instr_convs]
            inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]

            return torch.cat(inputs, 1)

        elif self.lang_model == 'bow':
            device = torch.device("cuda" if instr.is_cuda else "cpu")
            input_dim = self.obs_space["instr"]
            input = torch.zeros((instr.size(0), input_dim), device=device)
            idx = torch.arange(instr.size(0), dtype=torch.int64)
            input[idx.unsqueeze(1), instr] = 1.
            return self.instr_bow(input)
        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
