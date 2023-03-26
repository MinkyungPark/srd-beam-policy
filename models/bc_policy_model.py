import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model, 
    register_model_architecture,
    BaseFairseqModel
)
from fairseq.modules import FairseqDropout

from collections import OrderedDict


@register_model("bc_policy")
class BCPolicy(BaseFairseqModel):
    def __init__(self, embed_dim, inner_fn):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_act = 1
        self.hidden = self.embed_dim // 2
        self.inner_fn = inner_fn

        modules = OrderedDict()
        modules["linear"] = nn.Linear(self.embed_dim, self.hidden)
        modules["relu"] = nn.ReLU()
        
        for i in range(3):
            modules[f"linear_{i}"] = nn.Linear(self.hidden, self.hidden)
            modules[f"relu_{i}"] = nn.ReLU()
        modules["linear_4"] = nn.Linear(self.hidden, self.hidden // 2)
        modules["relu_4"] = nn.ReLU()
        modules["linear_outupt"] = nn.Linear(self.hidden // 2, self.num_act)

        self.layers = nn.Sequential(modules)
        
    @staticmethod
    def add_args(parser):
        parser.add_argument("--beam-decoding", type=bool, default=False)
        parser.add_argument("--max-len", type=int)

    @classmethod
    def build_model(cls, args, task):
        embed_dim = args.embed_dim
        inner_fn = task.inner_fn
        return cls(embed_dim, inner_fn)

    def forward(self, sample, **kwargs):
        if type(sample) == dict:
            inner_args = {
                "src_tokens": sample['src_tokens'],
                "src_lens": sample['src_lens'],
                "prefix_tokens": sample['prefix_tokens'],
                "device": sample['id'].device
            }
            mt_states = self.inner_fn(**inner_args)
        else:
            mt_states = sample[1]
        x = self.layers(mt_states)
        # Make Distributions
        dist = torch.distributions.Bernoulli(logits=x)
        # actions = dist.sample()
        return dist
    
    
@register_model("bc_policy_mixing")
class BCPolicyMixing(BaseFairseqModel):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_act = 1

        self.prev_lyr = nn.Linear(embed_dim, embed_dim)
        self.cur_lyr = nn.Linear(embed_dim, embed_dim)
        self.mixing_lyr = nn.Linear(embed_dim * 2, embed_dim)
        self.relu = nn.ReLU()

        modules = OrderedDict()
        for a, b in zip([1,1,2],[1,2,4]):
            modules[f"linear_{b}"] = nn.Linear(embed_dim // a, embed_dim // b)
            modules[f"relu_{b}"] = nn.ReLU()
        
        modules["linear_outupt"] = nn.Linear(self.embed_dim // 4, self.num_act)

        self.layers = nn.Sequential(modules)
        
    @staticmethod
    def add_args(parser):
        parser.add_argument("--beam-decoding", type=bool, default=False)
        parser.add_argument("--max-len", type=int)

    @classmethod
    def build_model(cls, args, task):
        embed_dim = args.embed_dim
        return cls(embed_dim)

    def forward(self, sample, **kwargs):
        if type(sample) == dict:
            mt_states = sample['source']
            prev_states = sample['prev_source']
        else:
            if sample[0] is None:
                prev_states = sample[1]
            else:
                prev_states = sample[0]
            mt_states = sample[1]
            
        prev = self.relu(self.prev_lyr(prev_states))
        cur = self.relu(self.cur_lyr(mt_states))
        cat_states = torch.concat((prev, cur), -1)
        mixed = self.relu(self.mixing_lyr(cat_states))
        
        x = self.layers(mixed)
        # Make Distributions
        dist = torch.distributions.Bernoulli(logits=x)
        # actions = dist.sample()
        return dist
    


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


@register_model("bc_policy_lstm")
class BCPolicyLSTM(BaseFairseqModel):
    def __init__(
        self, 
        embed_dim,
        hidden_size=512,
        dropout_in=0.1,
        dropout_out=0.1,
        num_layers=4,
        residuals=True,
        num_act=1,
    ):
        super().__init__()
        self.num_act = num_act
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.residuals = residuals
        
        self.src_proj = Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout_in_module = FairseqDropout(
            dropout_in * 1.0, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out * 1.0, module_name=self.__class__.__name__
        )
        
        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    input_size=embed_dim
                    if layer == 0
                    else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )
            
        self.out_proj_1 = Linear(hidden_size, hidden_size // 2)
        self.out_proj_2 = Linear(hidden_size // 2, self.num_act, dropout=dropout_out)
        
    @staticmethod
    def add_args(parser):
        parser.add_argument("--max-len", type=int)

    @classmethod
    def build_model(cls, args, task):
        embed_dim = args.embed_dim
        return cls(embed_dim)

    def forward(self, sample, **kwargs):
        if type(sample) == dict:
            mt_states = sample['source']
            pad_action_mask = kwargs["pad_action"].unsqueeze(-1)
            pad_action_mask = pad_action_mask.repeat(1, 1, self.hidden_size)
        else:
            mt_states = sample[1].view(1,1,-1)
            pad_action_mask=None
        
        bsz, tok_len, _ = mt_states.size()
        x = self.src_proj(mt_states)
        x = self.dropout_in_module(x)
        
        # B, T, D -> T, B, D
        x = x.transpose(0, 1)

        zero_state = x.new_zeros(bsz, self.hidden_size)
        prev_hiddens = [zero_state for i in range(self.num_layers)]
        prev_cells = [zero_state for i in range(self.num_layers)]
        outs = []

        for j in range(tok_len):
            input = x[j]
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell
        
            out = self.dropout_out_module(hidden)
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(tok_len, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        # delete pad action logit
        if pad_action_mask is not None:
            x = x[pad_action_mask].view(-1, self.hidden_size)
            x = x.view(-1, self.hidden_size)
        else:
            x = x.view(1,-1)
        x = self.out_proj_1(x)
        x = self.out_proj_2(x)
        
        # Make Distributions
        dist = torch.distributions.Bernoulli(logits=x)
        # actions = dist.sample()
        return dist
    
    
@register_model_architecture(model_name="bc_policy", arch_name="bc_policy")
def decision_policy(args):
    args.embed_dim = getattr(args, "embed_dim", 512)
    
@register_model_architecture(model_name="bc_policy_mixing", arch_name="bc_policy_mixing")
def decision_policy_mixing(args):
    args.embed_dim = getattr(args, "embed_dim", 512)
    
@register_model_architecture(model_name="bc_policy_lstm", arch_name="bc_policy_lstm")
def decision_policy_lstm(args):
    args.embed_dim = getattr(args, "embed_dim", 512)