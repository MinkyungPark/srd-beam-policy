# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class BCCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("bc_criterion", dataclass=BCCriterionConfig)
class BCCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.rnn = self.task.rnn

    def forward(self, model, sample, reduce=True):
        target_actions = sample['target']
        target_actions[target_actions == 2] = 1
        target_actions = target_actions.float()

        if self.rnn:
            pad_action = (target_actions != -1) # B, T
            target_actions = target_actions[pad_action]
            assert target_actions.size(0) == sample['net_input']['steps'].sum()
            dist = model(sample["net_input"], pad_action=pad_action)
            log_prob_actions = dist.log_prob(target_actions.view(-1, 1))
            loss = -(log_prob_actions).mean()
            sample_size = target_actions.size(0)
        else:
            dist = model(sample["net_input"])
            log_prob_actions = dist.log_prob(target_actions.view(-1, 1))
            loss = -(log_prob_actions).mean()
            sample_size = 1

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            # "sample_size": 1,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    


@register_criterion("bc_criterion_mse", dataclass=BCCriterionConfig)
class BCMSECriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.loss = nn.MSELoss()

    def forward(self, model, sample, reduce=True):
        dist = model(sample["net_input"])

        target_actions = sample['target']
        log_prob_actions = dist.log_prob(target_actions.view(-1, 1))
        loss = self.loss()
        loss = -(log_prob_actions).mean()

        sample_size = target_actions.size(0)
            
        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

