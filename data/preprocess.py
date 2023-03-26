import os
from pathlib import Path
from tqdm import tqdm

import math
import torch
import numpy as np

def rmtree(f: Path):
    if not f.exists():
        return
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()

def save_data(
    bucket_inds, 
    new_data, 
    data_path,
    save_path='preprocessed'
):
    save_path = data_path / save_path
    if not save_path.exists():
        save_path.mkdir()

    save_paths = []
    for i in range(len(bucket_inds)):
        sp = save_path / f'bucket_{i}'
        rmtree(sp)
        sp.mkdir()
        save_paths.append(sp)

    e_inds = torch.cumsum(
        torch.tensor(bucket_inds), dim=0
    ).long().tolist()
    s_inds = [0] + e_inds[:-1]

    for key, item in new_data.items():
        if key != 'inner_states':
            for i in range(len(bucket_inds)):
                s, e = s_inds[i], e_inds[i]
                save_item = item.clone()
                save_item = save_item[s:e].tolist()
                save_item = [str(it) for it in save_item]
                save_item = ' '.join(save_item)
                
                with open(save_paths[i] / 'data', 'a') as f:
                    f.write(key + '\n')
                    f.write(save_item + '\n')

    # for i in range(len(bucket_inds)):
    #     s, e = s_inds[i], e_inds[i]
    #     inner = new_data['inner_states'][s : e]

    #     path = save_path / f'bucket_{i}' / 'inner_states'
    #     np.save(path, inner)

def get_data_buckets(num_trans, div=12):
    total_num = torch.sum(num_trans)
    divided = math.ceil(total_num / div)
    num_trans = num_trans.tolist()

    bucket_nums, tmp = [], []
    tmp_flatten, appended = [], True
    for num_data in tqdm(num_trans, desc='load transitions for seperate bucket.'):
        tmp.append(num_data)
        if np.sum(tmp) > divided:
            bucket_nums.append(tmp)
            tmp = []
            appended = True
        else:
            appended = False
        tmp_flatten.append(num_data)

    if not appended:
        bucket_nums.append(tmp)
        
    bucket_inds = [np.sum(bk) for bk in bucket_nums]
    assert np.sum(bucket_inds) == total_num
    return bucket_inds

def concat_inner_states(
        data_path, 
        sorted_steps,
        ordered_sample_id
):
    inner_path = data_path / 'inner_states'
    num_traj = len(os.listdir(inner_path))
    assert sorted_steps.size(0) == num_traj

    concated = None
    for sent_step, sample_id in zip(
        tqdm(sorted_steps, desc='inner states merging.'),
        ordered_sample_id
    ):
        inner = torch.from_numpy(
            np.load(inner_path / f'inner_states_{sample_id}.npy', mmap_mode='c')
        )
        if concated is None:
            _, d = inner.size()
            concated = torch.zeros(1, d)

        concated = torch.concat(
            (concated, inner[1:sent_step+1]), dim=0
        )
    
    concated = concated[1:]
    assert concated.size(0) == torch.sum(sorted_steps)
    return concated

def preprocess(
    data_path='/workspace/data-bin/search_task/data'
):
    print(f'path : {data_path}')
    data_path = Path(data_path)
    data_keys = [x.name for x in data_path.iterdir() if not x.is_dir()]
    
    raw_data = {}
    for key in data_keys:
        with open(data_path / key, 'r') as f:
            datas = f.read().splitlines()
        raw_data[key] = []
        for data in datas:
            splited = data.split(':')
            _id, _data = int(splited[0]), splited[2]
            if key in ['al', 'bleus', 'delays', 'score']:
                tmp = (_id, [float(d) for d in _data.split()])
            else:
                tmp = (_id, [int(d) for d in _data.split()])
            raw_data[key].append(tmp)
        raw_data[key].sort(key=lambda x:x[0])
        print(f'Load done : {key}.')
    
    num_trajs, orderd_ids = [], []
    for key in data_keys:
        tmp, tmp_id = [], []
        for rd in raw_data[key]:
            tmp_id.append(rd[0])
            tmp.append(rd[1])
        orderd_ids.append(tmp_id)
        raw_data[key] = tmp
        num_trajs.append(len(raw_data[key]))
        
    assert torch.all(torch.tensor(num_trajs) == num_trajs[0])
    for orderd_id in orderd_ids:
        assert (orderd_ids[0] == orderd_id) == True
    
    # sort by sample id
    actions = raw_data["action_seqs"]
    steps = [len(act) for act in actions]
    max_step = np.max(steps)
    num_traj = len(actions)
    
    new_data = {}
    for key in data_keys:
        if key not in ['sample_id', 'score']:
            tmp = torch.full((num_traj, max_step), -1)
            tmp = tmp.float() if key in ['bleus', 'al'] else tmp.long()
            for i, step in enumerate(steps):
                tmp[i, :step] = torch.tensor(raw_data[key][i])
            new_data[key] = tmp
        else:
            tmp = torch.tensor(raw_data[key])
            new_data[key] = tmp.view(-1,1).repeat(1, max_step)

    # remove first step
    prefix_tokens = new_data['tokens'][:, :-1]
    src_idxs = new_data['src_idxs']
    new_data['src_idxs'] = torch.roll(src_idxs, 1, -1)

    for k in new_data.keys():
        new_data[k] = new_data[k][:, 1:]
    new_data['prefix_tokens'] = prefix_tokens
    data_keys.append('prefix_tokens')
    assert torch.all(new_data['tokens'][:, :-1] == new_data['prefix_tokens'][:, 1:])

    valid_mask = (new_data['action_seqs'] != -1)
    sent_steps = torch.sum(valid_mask, dim=-1)

    sent_steps, sorted_sent_inds = torch.sort(sent_steps, descending=True)

    valid_mask = valid_mask[sorted_sent_inds]

    for key in data_keys:
        sorted = new_data[key][sorted_sent_inds]
        new_data[key] = sorted[valid_mask]

    bucket_indices = get_data_buckets(sent_steps)
    new_data['sent_steps'] = torch.concat(
        [torch.arange(ss) for ss in sent_steps]
    )
    
    # ordered_sample_id = new_data['sample_id'][new_data['sent_steps'] == 0]
    # new_data['inner_states'] = concat_inner_states(
    #     data_path, sent_steps, ordered_sample_id
    # )

    save_data(bucket_indices, new_data, data_path)
    print('done.')

# preprocess('/workspace/data-bin/search_task/unidirectional_data_0.09')
# preprocess('/workspace/data-bin/search_task/unidirectional_data_0.15')
# preprocess('/workspace/data-bin/search_task/bidirectional_data_0.09')
# preprocess('/workspace/data-bin/search_task/bidirectional_data_0.15')