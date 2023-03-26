import os
import itertools
import logging
from pathlib import Path

import numpy as np
import torch

from fairseq.data import (
    FairseqDataset,
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)

logger = logging.getLogger(__name__)


class BCTrainDataset(FairseqDataset):
    def __init__(
        self,
        data,
        src_dict,
        tgt_dict=None,
        max_len=200,
        shuffle=True,
        rnn=False,
        mt_data=None,
    ):
        self.shuffle = shuffle
        self.rnn = rnn
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.max_len = max_len

        # B * T, D
        self.actions = data["action_seqs"]
        self.sample_ids = data["sample_ids"]
        self.sent_steps = data["sent_steps"]
        self.src_idxs = data["src_idxs"]
        self.indices = data["indices"]
        sent_steps = data["sent_steps"]
        # self.inners = inner_states.clone()

        prefix_tokens = data["prefix_tokens"]

        start_index = (sent_steps == 0).nonzero().view(-1)
        end_index = torch.concat(
            ((start_index - 1)[1:], torch.tensor([sent_steps.size(0) - 1]))
        )
        step_lens = sent_steps[end_index] + 1

        self.prefix_tokens = {} 
        for i, l in enumerate(step_lens.tolist()):
            sample_id = self.sample_ids[start_index[i]]
            p_toks = prefix_tokens[start_index[i] : end_index[i] + 1]
            self.prefix_tokens[sample_id.item()] = p_toks.tolist()

        # prev_inners = torch.concat(
        #     (inner_states[0:1], inner_states[:-1]), dim=0
        # )
        # prev_inners[start_index] = self.inners[start_index]
        # self.prev_inners = prev_inners
        
        self.src_sizes = np.array([1] * len(self))
        self.tgt_sizes = np.array([1] * len(self))

        if self.rnn:
            # B, T, D
            pad_step_len = step_lens.max().item()
            bsz = end_index.size(0)
            # inners = torch.ones(bsz, pad_step_len, self.inners.size(-1))
            # prev_inners = torch.ones(bsz, pad_step_len, self.inners.size(-1))
            actions = torch.full((bsz, pad_step_len), 2)
            for i, l in enumerate(step_lens.tolist()):
                # inners[i, :l] = self.inners[start_index[i] : end_index[i] + 1]
                # prev_inners[i, :l] = self.prev_inners[start_index[i] : end_index[i] + 1]
                actions[i, :l] = self.actions[start_index[i] : end_index[i] + 1]
            
            # self.inners = inners
            # self.prev_inners = prev_inners
            self.actions = actions

            self.sample_ids = self.sample_ids[end_index]
            self.sent_steps = step_lens

            self.indices = torch.arange(bsz)

            self.src_sizes = step_lens.numpy()
            self.tgt_sizes = step_lens.numpy()
        
        if mt_data is not None:
            self.mt_data = mt_data

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        step = self.sent_steps[index]
        action = self.actions[index]

        src_idx = self.src_idxs[index]
        source = self.mt_data[sample_id]['source'][:src_idx + 1]
        source = source[source != self.src_dict.pad()]
        src_token = source.tolist() + [self.src_dict.eos()]
        src_len = [len(src_token)]
        pad_token = torch.full((self.max_len,), self.src_dict.pad())
        pad_token[:len(src_token)] = torch.tensor(src_token).long()
        src_token = pad_token.tolist()
        
        prefix = torch.tensor(self.prefix_tokens[sample_id.item()])
        prefix = prefix[:step + 1][:self.max_len]
        prefix = prefix[(prefix != self.tgt_dict.pad())]
        pad_token = torch.full((self.max_len,), self.tgt_dict.pad())
        pad_token[0] = self.tgt_dict.eos() #########
        pad_token[1:prefix.size(0) + 1] = prefix
        # pad_token[0] = self.tgt_dict.eos()
        prefix_token = pad_token.tolist()

        example = {
            "net_input":{
                "id": sample_id,
                "src_tokens": src_token,
                "src_lens": src_len,
                "prefix_tokens": prefix_token,
                "steps": step,
            },
            "target": action,
        }

        return example

    def __len__(self):
        return len(self.indices)

    def collater(self, samples):
        if not samples:
            return
        
        ids, sources, prev_src = [], [], []
        targets, steps = [], []
        src_tokens, src_lens, prefix_tokens = [],[],[]
        for sample in samples:
            ids.append(sample["net_input"]["id"])
            # sources.append(sample["net_input"]["source"])
            steps.append(sample["net_input"]["steps"])
            src_tokens.append(sample["net_input"]["src_tokens"])
            src_lens.append(sample["net_input"]["src_lens"])
            prefix_tokens.append(sample["net_input"]["prefix_tokens"])
            targets.append(sample["target"])

        if not self.rnn:
            batch = {
                "net_input": {
                    "id": torch.LongTensor(ids),
                    # "source": torch.stack(sources),
                    "src_tokens": torch.LongTensor(src_tokens),
                    "src_lens": torch.LongTensor(src_lens),
                    "prefix_tokens": torch.LongTensor(prefix_tokens),
                    "steps": torch.LongTensor(steps),
                },
                "target": torch.LongTensor(targets)
            }
        else:
            steps = torch.LongTensor(steps)
            max_step = steps.max().item() + 1
            batch = {
                "net_input": {
                    "id": torch.LongTensor(ids),
                    # "source": torch.stack(sources)[:, :max_step],
                    "src_tokens": src_tokens,
                    "src_lens": src_lens,
                    "prefix_tokens": prefix_tokens,
                    "steps": steps,
                },
                "target": torch.stack(targets)[:, :max_step]
            }
        return batch

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.max_step

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.src_sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes, self.tgt_sizes)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        # if self.shuffle:
        #     indices = np.random.permutation(len(self)).astype(np.int64)
        # else:
        #     indices = np.arange(len(self), dtype=np.int64)
        return self.indices
            
    def prefetch(self, indices):
        return False
    
    def filter_indices_by_size(self, indices, max_sizes):
    
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes=None,
        )
    
def bc_dataset_load(data_path, device_num, device_id, div=12):
    bucket_inds = torch.arange(div).view(device_num, -1)
    cur_inds = bucket_inds[device_id]

    data_path = Path(data_path)
    data_path = data_path / 'preprocessed'

    # raw_data, raw_inners = None, []
    raw_data = None
    for i in cur_inds.tolist():
        bucket_path = data_path / f'bucket_{i}'
        with open(bucket_path / 'data', 'r') as f:
            raw = f.read().splitlines()
        data_keys = raw[::2]
        data_values = raw[1::2]
        if raw_data is None:
            raw_data = {k: v for k,v in zip(data_keys, data_values)}
        else:
            for k,v in zip(data_keys, data_values):
                raw_data[k] = ' '.join([raw_data[k], v])
        # inner_path = bucket_path / 'inner_states.npy'
        # inner = torch.from_numpy(
        #     np.load(inner_path, mmap_mode='c')
        # )
        # raw_inners.append(inner)
    
    sent_steps = torch.tensor(
        [int(step) for step in raw_data['sent_steps'].split()]
    ).long()
    # end_index = (sent_steps == 0).nonzero().view(-1) - 1
    # end_index = end_index[1:].tolist() + [sent_steps.size(0) - 1]
    # inners = torch.concat((raw_inners), dim=0)

    sample_ids = torch.tensor(
        [int(sample_id) for sample_id in raw_data['sample_id'].split()]
    ).long()
    
    action_seqs = torch.tensor(
        [int(act) for act in raw_data['action_seqs'].split()]
    ).long()

    src_idxs = torch.tensor(
        [int(act) for act in raw_data['src_idxs'].split()]
    ).long() 

    prefix_tokens = torch.tensor(
        [int(act) for act in raw_data['prefix_tokens'].split()]
    ).long()

    hyp_tokens = torch.tensor(
        [int(act) for act in raw_data['tokens'].split()]
    ).long()

    # Data check
    # assert inners.size(0) == sample_ids.size(0)
    assert sample_ids.size(0) == action_seqs.size(0)
    assert action_seqs.size(0) == sent_steps.size(0)

    assert torch.all(action_seqs == -1) == 0

    indices = torch.arange(sample_ids.size(0))

    # return sample_ids, inners, action_seqs, sent_steps, indices

    return {
        "sample_ids": sample_ids,
        "inners": None,
        "action_seqs": action_seqs,
        "sent_steps": sent_steps,
        "src_idxs": src_idxs,
        "prefix_tokens": prefix_tokens,
        "hyp_tokens": hyp_tokens,
        "indices": indices,
    }


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    save_path='',
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes.copy() if tgt_dataset is not None else None
    src_dataset_sizes = src_dataset.sizes.copy()

    save_path = Path(save_path)
    if save_path.exists():
        with open(save_path / 'sample_id', 'r') as f:
            done_ids = f.read().splitlines()
        done_ids = np.array(
            [int(d.split(':')[0]) for d in done_ids]
        )
        
        tgt_dataset_sizes.setflags(write=1)
        src_dataset_sizes.setflags(write=1)
        tgt_dataset_sizes[done_ids] = 10000
        src_dataset_sizes[done_ids] = 10000

    return LanguagePairDataset(
        src_dataset,
        src_dataset_sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )
