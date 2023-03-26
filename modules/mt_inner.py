import os
import torch

from fairseq import checkpoint_utils, tasks

def load_mt_model(model_path, return_dict=False):
    if not os.path.exists(model_path):
        raise IOError("Model file not found: {}".format(model_path))

    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)

    task_args = state["cfg"]["task"]
    task = tasks.setup_task(task_args)

    # build model for ensemble
    state["cfg"]["model"].load_pretrained_encoder_from = None
    state["cfg"]["model"].load_pretrained_decoder_from = None

    mt_model = task.build_model(state["cfg"]["model"])
    mt_model.load_state_dict(state["model"], strict=True)
    mt_model.eval()
    mt_model.share_memory()
    mt_model.cuda()

    print(f"[Translation Model] : {mt_model.args.arch}")

    if return_dict:
        return mt_model, task.target_dictionary, task.source_dictionary
    
    return mt_model


class MTInner():
    def __init__(self, model_path, pad_step=200):
        self.mt_model = load_mt_model(model_path)
        self.pad_step = pad_step

    @torch.no_grad()
    def get_inner(self, src_tokens, src_lens, prefix_tokens, device):
        model = self.mt_model
        model.to(device)
        out = model.decoder.forward(
            prev_output_tokens=prefix_tokens,
            encoder_out=model.encoder(src_tokens, src_lens)
        )
        out[1]['inner_states'][-1].detach()
        dec_state = out[1]['inner_states'][-1] # T(prefix), B, D
        inners = dec_state.permute(1,0,2) # B, T, D
        # 512 200 512
        index = (prefix_tokens != 1).sum(dim=-1) - 1
        index = index.view(-1,1,1).repeat(1,1,inners.size(-1))
        inners = torch.gather(
            inners, 1, index=index
        ).squeeze()
        torch.cuda.empty_cache()
        return inners
