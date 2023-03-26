# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gc
import copy
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from sacrebleu import sentence_bleu

from .search import ValueSearch
from .beam_decoding import BeamDecoding
from ..modules.latency import length_adaptive_average_lagging


class BeamPolicyGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        eos=None,
        symbols_to_strip_from_output=None,
        tokens_to_suppress=(),
        data_round=0,
        al_const=0.2,
        pre_model=None,
        encoder_type="unidirectional",
        save_path='',
        **kwargs,
    ):
        super().__init__()
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        
        self.decode = BeamDecoding(
            models=models,
            tgt_dict=tgt_dict,
            beam_size=beam_size,
            vocab_size=len(tgt_dict),
            tokens_to_suppress=tokens_to_suppress,
            symbols_to_strip_from_output=symbols_to_strip_from_output,
            normalize_scores=normalize_scores,
            len_penalty=len_penalty,
            unk_penalty=unk_penalty,
            temperature=temperature,
            match_source_len=match_source_len,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        self.enc_embed_dim = self.decode.enc_embed_dim

        self.al_const = al_const
        self.enc_type = encoder_type

        self.round = data_round
        self.set_save_path(save_path)
        self.value_search = ValueSearch(tgt_dict)
    
    def set_device(self, device):
        device = device
        self.decode.set_device(device)

    def set_save_path(self, path):
        path = Path(path)
        self.top_path = path / f'{self.enc_type}_data_{self.al_const}'        
        self.replay_path = self.top_path / 'replay'
        self.replay_inner_path = self.replay_path / 'inner_states'
        self.top_inner_path = self.top_path / 'inner_states'

        if not self.top_path.exists():
            self.top_inner_path.mkdir(parents=True)
            self.replay_inner_path.mkdir(parents=True)

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:

        result = self._generate(sample, **kwargs)
        return result
    
# ------------------------------------------ Set Encoder State
    def get_encoder_state(self, src_len, bsz, src_tokens, device):
        encoder = self.decode.model.single_model.encoder
        if self.enc_type == 'unidirectional':
            enc_incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]],
                    torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
            )
        else: # Bidirectional
            enc_incremental_state = None
            without_eos_incremental = None

        total_enc_out = torch.ones(src_len, bsz, (src_len + 1) * self.enc_embed_dim)
        total_enc_out = total_enc_out.to(device).float()
        total_enc_pad_mask = torch.ones(src_len, bsz, (src_len + 1))
        total_enc_pad_mask = total_enc_pad_mask.to(device).bool()
        
        for i in range(src_len):
            stream_src = src_tokens[:, :i+1]
            eos_or_pad = torch.all(
                            (stream_src.ne(self.eos) & stream_src.ne(self.pad))
                            , dim=1
                        ).unsqueeze(1) # eos나 pad index
            padding = torch.full((bsz,1), self.eos).to(src_tokens)
            padding = padding.masked_fill_(~eos_or_pad, self.pad)
            stream_src_eos = torch.concat([stream_src, padding], dim=1)

            if self.enc_type == 'unidirectional':            
                # without eos
                stream_net_input = {
                    'src_tokens': stream_src,
                    'incremental_state': enc_incremental_state
                }
                
                with torch.autograd.profiler.record_function("forward_encoder"):
                    encoder.forward_torchscript(stream_net_input)
                    
                without_eos_incremental = copy.deepcopy(enc_incremental_state)
            
            # with eos
            net_input_eos = {
                'src_tokens': stream_src_eos,
                'incremental_state': enc_incremental_state
            }
            
            with torch.autograd.profiler.record_function("forward_encoder"):
                eos_enc_out = encoder.forward_torchscript(net_input_eos)
                
                enc_out = eos_enc_out['encoder_out'][0] # L, B, D
                enc_pad = eos_enc_out['encoder_padding_mask'][0] # B, L

                # L, B, D -> B, L, D
                enc_out = enc_out.permute(1,0,2).contiguous() 
                enc_out = enc_out.view(bsz,-1) # B, LL*D
                total_enc_out[i, :, :enc_out.size(-1)] = enc_out
                total_enc_pad_mask[i, :, :enc_pad.size(-1)] = enc_pad
                
            enc_incremental_state = without_eos_incremental

        encoder_outs = [{k:[] for k in eos_enc_out.keys()}]
        
        # Reshape enc states
        # L, B, LL*D -> B, L, LL*D
        total_enc_out = total_enc_out.permute(1, 0, -1).contiguous()
        total_enc_out = total_enc_out.view(bsz, src_len, -1)
        # L, B, LL -> B, L, LL
        total_enc_pad_mask = total_enc_pad_mask.permute(1, 0, 2).contiguous()
        
        b, l, _ = total_enc_out.size()
        total_enc_out = total_enc_out.view(b, l, l+1, self.enc_embed_dim)
        return total_enc_out.cpu(), total_enc_pad_mask.cpu(), encoder_outs
    
    def get_prefix(self, hyp_tokens, step):
        prev_tokens = hyp_tokens[:, :step+1]
        prev_mask = (prev_tokens != self.pad)
        
        mask_sum = torch.cumsum(prev_mask, dim=1)
        prev_idxs = torch.masked_fill(mask_sum, ~prev_mask, 0)

        prev_tokens = torch.scatter(
            torch.ones_like(prev_tokens), dim=1,
            index=prev_idxs, src=prev_tokens
        )[:, 1:]
        
        return prev_tokens
# ------------------------------------------ Generate
    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        read_size: int,
        search_size: int,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
        post_process=None,
        decode_fn=None,
    ):
        src_tokens = sample["net_input"]["src_tokens"]
        device = src_tokens.device
        self.set_device(device)

        bsz, seq_len = src_tokens.size()[:2]
        tgt_len = sample['target'].size(1)
        max_step = (seq_len + tgt_len) * 2

        # Origin : bsz
        origin = {
            "ids": sample["id"],
            "src_toks": src_tokens,
            "src_lens": sample["net_input"]["src_lengths"],
            "bth_idxs": torch.arange(bsz).to(sample["id"]),
            "targets": sample['target']
        }

        # Prepare Encoder Out
        total_enc_out, total_enc_pad_mask, encoder_outs = self.get_encoder_state(
            seq_len, bsz, origin["src_toks"], device
        )
        # Repeat for search/read size
        ss_bsz = bsz * search_size

        # Finalized : bsz * search_size sorted by scores
        final = {
            k: torch.zeros(ss_bsz, max_step + 1).long().to(device)
            for k in [
                "src_idxs", "read_finish", "finish",
                "action_seqs", "hypo_tokens",
                "delays", "al", "bleus", "scores", 
            ]
        }
        final["action_seqs"][:, 0] = 1 # for first read token
        final["action_seqs"][:, 1:] = -1 # action pad
        final["finish"][:, 1:] = 1 # finish pad
        final["hypo_tokens"] += self.pad
        for float_key in ['delays', 'al', 'bleus', 'scores']:
            final[float_key] = final[float_key].float()
        final["inner_states"]: Tensor = None

        ss_bth_idxs = torch.repeat_interleave(
            origin["bth_idxs"], search_size, 0
        )
        search_idxs = torch.arange(ss_bsz).to(device)
        ll, d = seq_len + 1, self.enc_embed_dim 
# ------------------------------------------ Node Search
        for step in tqdm(range(max_step)):
            finish_mask = final['finish'].bool()[:, step]
            if finish_mask.sum() > 0:
                print()
            if torch.all(finish_mask):
                break
            
            # active : search size
            atv_mask = ~finish_mask
            atv_ss_bsz = torch.sum(atv_mask).item()
            
            atv_bth_idxs = ss_bth_idxs[atv_mask]
            atv_cur_idxs = final["src_idxs"][atv_mask, step]

            # w, rw, rrw .. src idx
            cum = torch.tensor(
                [[1] * (read_size-1)]
            ).repeat(atv_ss_bsz, 1).to(device)
            atv_rs_cur_idxs = torch.cumsum(
                torch.concat(
                    (atv_cur_idxs.view(-1,1), cum), -1
                ), -1
            )
            # Remove over src_len
            # active read size
            # : (search_size * read_size) - overlen sent
            atv_seq_len = torch.index_select(
                origin["src_lens"], 0, atv_bth_idxs
            ).view(-1,1)
            atv_seq_len = atv_seq_len.repeat(1, read_size)
            atv_rs_bth_idxs = atv_bth_idxs.view(-1,1).repeat(1, read_size)
            
            over_idx_mask = atv_seq_len > atv_rs_cur_idxs

            atv_actions = torch.arange(read_size).to(device)
            atv_actions = atv_actions.repeat(atv_ss_bsz, 1)
            atv_actions = atv_actions[over_idx_mask].view(-1)

            atv_search_idxs = search_idxs[atv_mask].view(-1,1)
            atv_search_idxs = atv_search_idxs.repeat(1, read_size)
            atv_search_idxs = atv_search_idxs[over_idx_mask].view(-1)

            atv_rs_cur_idxs = atv_rs_cur_idxs[over_idx_mask].view(-1).cpu()
            atv_rs_bth_idxs = atv_rs_bth_idxs[over_idx_mask].view(-1).cpu()

            """
            total_enc_out B x L x LL * D
            total_enc_pad_mask B x L x LL
            """

            rs_enc_outs = total_enc_out[atv_rs_bth_idxs]
            gather_idx = atv_rs_cur_idxs.view(-1,1,1,1)
            gather_idx = gather_idx.repeat(1,1,ll,d)
            rs_enc_outs = torch.gather(
                rs_enc_outs, 1, gather_idx
            ).view(-1,ll,d).to(device)

            rs_pad_masks = total_enc_pad_mask[atv_rs_bth_idxs]
            gather_idx = gather_idx[:,:,:,0]
            rs_pad_masks = torch.gather(
                rs_pad_masks, 1, gather_idx
            ).view(-1,ll).to(device)
            
            atv_rs_bth_idxs.to(device)
            atv_rs_cur_idxs.to(device)
            a,b,c = rs_enc_outs.size()
            assert a == over_idx_mask.view(-1).sum()
            assert (b == ll) and (c == d)

            # atv_rsb, ll, d -> ll, atv_rsb, d
            rs_enc_outs = rs_enc_outs.permute(1, 0, 2).contiguous()
            encoder_outs[0]['encoder_out'] = [rs_enc_outs]
            # atv_rsb, ll
            encoder_outs[0]['encoder_padding_mask'] = [rs_pad_masks]

            prev_hypos = final['hypo_tokens'][atv_search_idxs]
            # Prefix tokens
            prefix_tokens = self.get_prefix(
                prev_hypos, step
            )
                
            finalized = self.decode.decode(
                encoder_outs,
                prefix_tokens=prefix_tokens,
            )

            if final["inner_states"] is None:
                _inner = finalized[0][0]["inners"]
                dim = _inner.size(-1)
                final["inner_states"] = torch.zeros(
                    ss_bsz, max_step + 1, dim
                ).to(_inner)
            
# ------------------------------------------ Scores
            item_keys = ['hypo_tokens', 'bleus', 'al', 'delays', 'scores']
            score_result = {
                k: torch.empty_like(atv_actions).float().to(device)
                for k in item_keys
            }
            score_result['hypo_tokens'] = score_result['hypo_tokens'].long()
            score_result['inner_states'] = torch.empty(
                atv_actions.size(0), d
            ).to(device)

            assert len(finalized) == atv_rs_bth_idxs.size(0)
            assert atv_rs_bth_idxs.size(0) == atv_search_idxs.size(0)
            assert atv_search_idxs.size(0) == atv_actions.size(0)

            for i in range(len(finalized)):
                cur_bth_ind = atv_rs_bth_idxs[i] # origin batch idx
                cur_search_bth_ind = atv_search_idxs[i] # search batch idx
                cur_action = atv_actions[i]
                elem = finalized[i]
                cur_src_ind = atv_rs_cur_idxs[i]

                target_str = decode_fn(
                    self.tgt_dict.string(
                        origin["targets"][cur_bth_ind],
                        post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore={self.eos, self.pad}
                    )
                )
                prev_tok = prefix_tokens[i]
                hyp_step = torch.sum(
                    (prev_tok.ne(self.pad)), dim=-1
                ).item()
                # hyp_step = hyp_step.item()
                beam_tok = elem[0]['tokens'][:(hyp_step + 1 + cur_action)]
                assert torch.all(
                    prev_tok[:hyp_step] == beam_tok[:hyp_step]
                )
                hyp_str = self.tgt_dict.string(
                    beam_tok.tolist(),
                    post_process,
                    escape_unk=True,
                    extra_symbols_to_ignore={self.pad}
                )

                b = sentence_bleu(decode_fn(hyp_str), [target_str])
                bleu_score = torch.tensor(round(b.score, 2))
                
                cur_src_len = origin['src_lens'][cur_bth_ind]
                cur_tgt_len = torch.tensor(
                    origin['targets'][cur_bth_ind].size(0)
                ).to(device)

                delay = torch.min(
                    (cur_src_ind + 1), cur_src_len
                ).view(1)

                prev_delay = final['delays'][cur_search_bth_ind][:step+1]
                write_mask = (final['action_seqs'][cur_search_bth_ind][:step+1] == 0)
                prev_delay = prev_delay[write_mask]

                all_delay = torch.concat((prev_delay, delay), -1)
                oracle = torch.tensor(
                    all_delay.size(0)
                ).to(device)

                al = length_adaptive_average_lagging(
                    all_delay.view(1,-1), # delay
                    cur_src_len.view(1,1), # src len
                    oracle.view(1,1), # target(oracle)
                    cur_tgt_len.view(1,1), # reference
                )
                if al < 0:
                    al = 50

                score = bleu_score - (self.al_const * step * al)
                # score = bleu_score - (0.05 * step * al)
                # score = bleu_score - (self.al_const * al)

                hyp_token = beam_tok[hyp_step]
                if cur_action > 0: # read
                    hyp_token = torch.tensor(self.pad).long()
                    # delay = torch.min(
                    #     (cur_src_ind + 1 - cur_action), cur_src_len
                    # ).view(1)
                    elem = finalized[i - cur_action]

                score_result['hypo_tokens'][i] = hyp_token
                score_result['bleus'][i] = bleu_score
                score_result['al'][i] = al
                score_result['delays'][i] = delay
                score_result['scores'][i] = score
                score_result['inner_states'][i] = elem[0]['inners'][hyp_step]

            final = self.value_search.step(
                final=final, 
                atv_actions=atv_actions,
                atv_search_idxs=atv_search_idxs,
                atv_rs_bth_idxs=atv_rs_bth_idxs,
                score_result=score_result, 
                step=step,
                read_size=read_size,
                keep_size=search_size,
                batch_finish=finish_mask,
                search_bsz=ss_bsz,
                ss_bth_idxs=ss_bth_idxs,
                search_idxs=search_idxs,
                src_lenghts=origin['src_lens'],
            )

        # Finish Search
        result = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )
        for s_idx in range(0, ss_bsz, search_size):
            s, e = s_idx, s_idx + search_size
            bth_inds = ss_bth_idxs[s:e]
            assert torch.all(bth_inds == bth_inds[0].item())
            bth_ind = bth_inds[0]
            actions = final['action_seqs'][s:e]
            
            finish_steps = torch.sum((actions != -1), dim=1) - 1
            scores = final['scores'][s:e]
            scores = torch.gather(
                scores, dim=1,
                index=finish_steps.unsqueeze(1)
            ).squeeze()
            _, sorted_scores_indices = torch.sort(scores, descending=True)

            src_idxs = final['src_idxs'][s:e]
            hypo_tokens = final['hypo_tokens'][s:e]
            als = final['al'][s:e]
            delays = final['delays'][s:e]
            bleus = final['bleus'][s:e]
            read_finish = final['read_finish'][s:e]
            finish = final['finish'][s:e]
            inners = final['inner_states'][s:e]

            _id = origin["ids"][bth_ind].item()


            result[bth_inds[0]] = [
                {
                    "src_idxs": src_idxs[ssi].cpu(),
                    "action_seqs": actions[ssi].cpu(),
                    "tokens": hypo_tokens[ssi].cpu(),
                    "al": als[ssi].cpu(),
                    "bleus": bleus[ssi].cpu(),
                    "delays": delays[ssi].cpu(),
                    "score": scores[ssi].cpu().unsqueeze(0), # 
                    "read_finish": read_finish[ssi].cpu(),
                    "finish": finish[ssi].cpu(),
                    "inner_states": inners[ssi].cpu(),
                    "alignment": torch.empty(0),
                    "sample_id": torch.tensor([_id]).long(), #
                } for ssi in sorted_scores_indices
            ]
        
        result_keys = list(result[0][0].keys())
        result_keys.remove("inner_states")
        result_keys.remove("alignment")
        for res in result:
            # top save
            best_res = res[0]
            sample_id = best_res['sample_id'].item()
            for key in result_keys:
                item = best_res[key]
                with open(self.top_path / key, 'a') as f:
                    string = ' '.join([str(it) for it in item.tolist()])
                    f.write(f'{sample_id}:0:' + string + '\n')
            inner_path = self.top_inner_path / f'inner_states_{sample_id}'
            np.save(inner_path, best_res['inner_states'])

            # other save
            for i, other_res in enumerate(res[1:]):
                for key in result_keys:
                    item = other_res[key]
                    with open(self.replay_path / key, 'a') as f:
                        string = ' '.join([str(it) for it in item.tolist()])
                        f.write(f'{sample_id}:{i+1}:' + string + '\n')
                inner_path = self.replay_inner_path / f'inner_states_{sample_id}_{i+1}'
                np.save(inner_path, other_res['inner_states'])

        gc.collect()
        torch.cuda.empty_cache()
        return result
    

