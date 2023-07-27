import torch
import numpy as np
import utils.pooling as POOLING
from torch import nn
from utils import loss as L
from torch.nn import functional as F
from utils.rnns import gather_last
from utils.rnns import feed_forward_rnn
from solver import make_optimizer, make_scheduler
from modeling.language.build import get_language_model

class ExCL(nn.Module):
    def __init__(self, cfg):
        super(ExCL, self).__init__()

        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE_TRAIN

        if self.cfg.LANGUAGE.MODEL != "glove":
            self.language_model = get_language_model(self.cfg)

        self.language_lstm = nn.LSTM(
            input_size=cfg.LANGUAGE_LSTM.INPUT_SIZE,
            hidden_size=cfg.LANGUAGE_LSTM.HIDDEN_SIZE,
            num_layers=cfg.LANGUAGE_LSTM.NUM_LAYERS,
            bias=cfg.LANGUAGE_LSTM.BIAS,
            dropout=cfg.LANGUAGE_LSTM.DROPOUT,
            bidirectional=cfg.LANGUAGE_LSTM.BIDIRECTIONAL,
            batch_first=cfg.LANGUAGE_LSTM.BATCH_FIRST,
        )

        self.video_lstm = nn.LSTM(
            input_size=cfg.VIDEO_LSTM.INPUT_SIZE,
            hidden_size=cfg.VIDEO_LSTM.HIDDEN_SIZE,
            num_layers=cfg.VIDEO_LSTM.NUM_LAYERS,
            bias=cfg.VIDEO_LSTM.BIAS,
            dropout=cfg.VIDEO_LSTM.DROPOUT,
            bidirectional=cfg.VIDEO_LSTM.BIDIRECTIONAL,
            batch_first=cfg.VIDEO_LSTM.BATCH_FIRST,
        )

        self.localization_lstm = nn.LSTM(
            input_size=cfg.LOCALIZATION.INPUT_SIZE,
            hidden_size=cfg.LOCALIZATION.HIDDEN_SIZE,
            num_layers=cfg.LOCALIZATION.NUM_LAYERS,
            bias=cfg.LOCALIZATION.BIAS,
            dropout=cfg.LOCALIZATION.DROPOUT,
            bidirectional=cfg.LOCALIZATION.BIDIRECTIONAL,
            batch_first=cfg.LOCALIZATION.BATCH_FIRST,
        )

        self.starting = nn.Sequential(
            nn.Linear(cfg.CLASSIFICATION.INPUT_SIZE, 256),
            nn.Tanh(),
            nn.Linear(256, cfg.CLASSIFICATION.OUTPUT_SIZE),
        )
        self.ending = nn.Sequential(
            nn.Linear(cfg.CLASSIFICATION.INPUT_SIZE, 256),
            nn.Tanh(),
            nn.Linear(256, cfg.CLASSIFICATION.OUTPUT_SIZE),
        )
        self.dropout = nn.Dropout(p=0.5)

    def attention(self, videoFeat, filter, lengths):
        pred_local = torch.bmm(videoFeat, filter.unsqueeze(2)).squeeze()
        return pred_local

    def get_mask_from_sequence_lengths(
        self, sequence_lengths: torch.Tensor, max_length: int
    ):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def masked_softmax(
        self,
        vector: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
        memory_efficient: bool = False,
        mask_fill_value: float = -1e32,
    ):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill(
                    (1 - mask).byte(), mask_fill_value
                )
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

    def mask_softmax(self, feat, mask):
        return self.masked_softmax(feat, mask, memory_efficient=False)

    def kl_div(self, p, gt, length):
        individual_loss = []
        for i in range(length.size(0)):
            vlength = int(length[i])
            ret = gt[i][:vlength] * torch.log(p[i][:vlength] / gt[i][:vlength])
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

    def nnlog(self, p_start, p_end, gt_start, gt_end, length):
        indx_start = torch.argmax(gt_start, dim=1)
        indx_end = torch.argmax(gt_end, dim=1)
        aux = 0
        for i in range(p_start.shape[0]):
            # print(p_start[i])
            aux += torch.log(p_start[i][indx_start[i]]) + torch.log(
                p_end[i][indx_end[i]]
            )
        return -torch.mean(aux)
    
    def forward(
        self,
        video_info,
        query_info,
        start,
        end,
        localiz,
    ):
        videoFeat = video_info["videoFeat"]
        videoFeat_lengths = video_info["videoFeat_lengths"]

        if self.cfg.LANGUAGE.MODEL == "glove":
            output_language, _ = feed_forward_rnn(
                self.language_lstm, query_info["tokens"], lengths = query_info["tokens_lengths"]
            )

        else:
            bert_output = self.language_model(
                input_ids      = query_info["tokens_ids"], 
                position_ids   = query_info["position_ids"], 
                attention_mask = query_info["lang_mask"], output_hidden_states=True
            )

            sequences = bert_output.last_hidden_state

            output_language, _ = feed_forward_rnn(
                self.language_lstm, sequences, lengths = query_info["tokens_lengths"]
            )

        output_language = gather_last(output_language, query_info["tokens_lengths"].long())
        output_video, _ = feed_forward_rnn(
            self.video_lstm, videoFeat, videoFeat_lengths
        )
        input_localization = torch.cat(
            (
                output_language.unsqueeze(1).repeat(
                    1, output_video.shape[1], 1
                ),
                output_video,
            ),
            dim=2,
        )
        # print(F.dropout(input_localization, p=0.1))
        output_localization, _ = feed_forward_rnn(
            self.localization_lstm,
            input_localization,
            videoFeat_lengths,
        )
        mask = self.get_mask_from_sequence_lengths(
            videoFeat_lengths, int(videoFeat.shape[1])
        )

        final_output = torch.cat(
            (
                output_language.unsqueeze(1).repeat(
                    1, output_video.shape[1], 1
                ),
                output_video,
                input_localization,
            ),
            dim=2,
        )
        pred_start = (
            self.starting(final_output.view(-1, final_output.size(2)))
            .view(-1, final_output.size(1), 1)
            .squeeze()
        )
        pred_start = self.mask_softmax(pred_start, mask)

        pred_end = (
            self.ending(final_output.view(-1, final_output.size(2)))
            .view(-1, final_output.size(1), 1)
            .squeeze()
        )
        pred_end = self.mask_softmax(pred_end, mask)

        start_loss, individual_start_loss = self.kl_div(
            pred_start, start, videoFeat_lengths
        )
        end_loss, individual_end_loss = self.kl_div(
            pred_end, end, videoFeat_lengths
        )
        individual_loss = individual_start_loss + individual_end_loss

        total_loss = self.nnlog(
            pred_start, pred_end, start, end, videoFeat_lengths
        )

        return (
            total_loss,
            individual_loss,
            pred_start,
            pred_end,
            torch.zeros(len(videoFeat), int(max(videoFeat_lengths))),
            torch.tensor(0)
        ) 