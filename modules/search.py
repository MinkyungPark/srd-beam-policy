import math

import numpy as np
import torch
from torch import Tensor, BoolTensor, LongTensor
from typing import List, Dict, Optional

from fairseq.search import Search


class ValueSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.constraint_states = None
        self.policy = [0.5, 0.5]
        self.num_act = 2

        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()

    @torch.jit.export
    def step(
        self,
        final: Dict[str, Tensor], # {ss}
        atv_actions: LongTensor, # (active_rs,)
        atv_search_idxs: LongTensor, # batch search idx / (active_rs,)
        atv_rs_bth_idxs: LongTensor, # origin batch idx / (active_rs,)
        score_result: Dict[str, Tensor], # w, rw, rrw.. result / {active_rs}
        step: int,
        read_size: int,
        keep_size: int,
        batch_finish: BoolTensor, # ss_bsz
        search_bsz: int,
        ss_bth_idxs: int, # 0000 1111 ... / (ss,)
        search_idxs: int, # 0123 4567 ... / (ss,)
        src_lenghts: LongTensor,
    ):
        # action over 1 -> read (1)
        # atv_actions[atv_actions > 1] = 1
        device = atv_actions.device
        
        for si in range(0, search_bsz, keep_size):
            s, e = si, (si + keep_size) # not active, all !!
            
            # Finished final result into topk candidates
            finished_mask = batch_finish[s:e]
            if torch.all(finished_mask):
                # Copy step -> step + 1
                for k in final.keys():
                    final[k][s:e, step + 1] = final[k][s:e, step]
                continue
                    
            finish_idx = search_idxs[s:e][finished_mask]
            finish_num = finish_idx.size(0)
            
            batch_idx = ss_bth_idxs[s:e]
            assert torch.all(batch_idx == batch_idx[0])
            batch_idx = batch_idx[0].item()
            atv_mask = (atv_rs_bth_idxs == batch_idx)
            
            active = {}
            for k, v in score_result.items():
                # Take the step value from finished batch
                finished = final[k][s:e, step][finished_mask]
                # Concat actvie batch, finish batch
                active[k] = torch.concat(
                    (v[atv_mask], finished), 0
                )
            
            finish_actions = torch.full((finish_num,), -1).to(device)
            cur_actions = torch.concat(
                (atv_actions[atv_mask], finish_actions), -1
            )
            score_idx = torch.concat(
                (atv_search_idxs[atv_mask], finish_idx), -1
            )
            assert score_idx.numel() == active['scores'].numel()
            
            unique_scores = torch.unique(
                active['scores'], return_inverse=True
            )[0]
            if unique_scores.size(0) < keep_size:
                unique_scores = unique_scores.sort(descending=True).values
                unique_scores = unique_scores.repeat(keep_size)
                unique_scores = unique_scores[:keep_size]

            top_scores, top_indices = torch.topk(
                unique_scores,
                k=keep_size
            )
# ------------------------------------------ Random action for diverse search
            for i, ts in enumerate(top_scores):
                top_mask = (active['scores'] == ts)
                top_idx = top_mask.nonzero().view(-1)
                if top_mask.sum(0) > 1:
                    top_idx = np.random.choice(top_idx.tolist())
                top_indices[i] = top_idx

            prev_idx = score_idx[top_indices]
# ------------------------------------------ Update Topk
            # previous set up
            for k in final.keys():
                final[k][s:e, :step+1] = torch.index_select(
                    final[k], 0, prev_idx
                )[:, :step+1]
                
            # Update Action
            final['action_seqs'][s:e, step+1] = cur_actions[top_indices]
            
            # Update hypo, bleu, delay, score, inner
            for k in active.keys():
                final[k][s:e, step+1] = active[k][top_indices]

            # Update src_idx, read & write finish
            read = (final['action_seqs'][s:e, step+1] > 0).long()
            src_idxs = final['src_idxs'][s:e, step] + read
            final['src_idxs'][s:e, step + 1] = src_idxs
            
            cur_src_len = src_lenghts[batch_idx]
            read_finish = ((src_idxs + 1) == cur_src_len).long()
            assert torch.all((src_idxs + 1) <= cur_src_len)
            final['read_finish'][s:e, step + 1] = read_finish.long()
            
            hyp_tokens = final['hypo_tokens'][s:e, step + 1]
            final['hypo_tokens'][s:e, step + 1] = hyp_tokens
            finish = (hyp_tokens == self.eos).long()
            # ~read_finish & eos -> pad X
            final['finish'][s:e, step + 1] = finish

        return final
    

class BeamSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.constraint_states = None

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores: Optional[Tensor],
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        # lprobs (1, B, detokens)
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = torch.div(indices_buf, vocab_size, rounding_mode="trunc")
        indices_buf = indices_buf.fmod(vocab_size)

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf


class PrefixConstrainedBeamSearch(Search):
    def __init__(self, tgt_dict, prefix_allowed_tokens_fn):
        super().__init__(tgt_dict)
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.stop_on_max_len = True

    @torch.jit.export
    def apply_mask(self, x, prev_output_tokens, original_batch_idxs):
        beam_size = x.shape[0] // original_batch_idxs.shape[0]
        original_batch_idxs = (
            original_batch_idxs.unsqueeze(-1).repeat((1, beam_size)).flatten().tolist()
        )

        mask = torch.full_like(x, -math.inf)
        for sent_i, (sent, batch_i) in enumerate(
            zip(prev_output_tokens, original_batch_idxs)
        ):
            mask[sent_i, :, self.prefix_allowed_tokens_fn(batch_i, sent)] = 0

        return mask

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs: Tensor,
        scores: Tensor,
        prev_output_tokens: Tensor,
        original_batch_idxs: Tensor,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        lprobs += self.apply_mask(
            lprobs.view(bsz * beam_size, 1, vocab_size),
            prev_output_tokens,
            original_batch_idxs,
        ).view(bsz, beam_size, vocab_size)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf