from torch import nn
from torch import functional
import torch
from utils import sequence_mask
from hyper_parameters import tacotron_params


# from https://github.com/mozilla/TTS
class GuidedAttentionLoss(torch.nn.Module):
    def __init__(self, sigma=0.05):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma  # We need to experiment with this sigma value (thickness of the diagonal mask)

    def _make_ga_masks(self, ilens, olens):
        B = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ga_masks[idx, :olen, :ilen] = self._make_ga_mask(ilen, olen, self.sigma)
        return ga_masks

    def forward(self, att_ws, ilens, olens):
        ga_masks = self._make_ga_masks(ilens, olens).to(att_ws.device)
        seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = ga_masks * att_ws
        loss = torch.mean(losses.masked_select(seq_masks))
        return loss

    @staticmethod
    def _make_ga_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen, device=olen.device), torch.arange(ilen, device=ilen.device))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = sequence_mask(ilens)
        out_masks = sequence_mask(olens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


class MSELossMasked(nn.Module):

    def __init__(self, seq_len_norm):
        super(MSELossMasked, self).__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            # loss = functional.mse_loss(
            #     x * mask, target * mask, reduction='none')
            loss = nn.MSELoss(reduction='none')(x * mask, target * mask)
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            # loss = functional.mse_loss(
            #     x * mask, target * mask, reduction='sum')
            loss = nn.MSELoss(reduction='sum')(x * mask, target * mask)
            loss = loss / mask.sum()
        return loss


class BCELossMasked(nn.Module):

    def __init__(self, pos_weight):
        super(BCELossMasked, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length, max_len=target.size(1)).float()
        # loss = functional.binary_cross_entropy_with_logits(
        #     x * mask, target * mask, pos_weight=self.pos_weight, reduction='sum')
        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='sum')(x * mask, target * mask)
        loss = loss / mask.sum()
        return loss


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, input_lens, iteration):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        # Ensures dimension 1 will be size 1, the rest can be adapted. It is a column of length 189 with all zeroes
        # till the end of the current sequence, which is filled with 1's
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignment_fine, mel_out_coarse, alignment_coarse, out_lengths = model_output
        # Mean Square Error (L2) loss function for decoder generation + post net generation
        # mel_loss = nn.MSELoss()(mel_out, mel_target) + \
        #     nn.MSELoss()(mel_out_postnet, mel_target)
        mel_loss = MSELossMasked(tacotron_params['seq_len_norm'])(mel_out, mel_target, out_lengths) \
                   + MSELossMasked(tacotron_params['seq_len_norm'])(mel_out_postnet, mel_target, out_lengths)
        # Binary Cross Entropy with a Sigmoid layer combined. It is more efficient than using a plain Sigmoid
        # followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp
        # trick for numerical stability
        gate_out = gate_out.view(-1, 1)
        gate_loss = BCELossMasked(pos_weight=torch.tensor(10))(gate_out, gate_target, out_lengths)
        # gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        # LOSS OF DDC:
        mel_out_coarse = mel_out_coarse.transpose(1, 2)
        # decoder_coarse_loss = nn.MSELoss()(mel_out_coarse, mel_target)  # I need output lengths out_lengths
        decoder_coarse_loss = MSELossMasked(tacotron_params['seq_len_norm'])(mel_out_coarse, mel_target, out_lengths)
        # ATTENTION LOSS:
        attention_loss = nn.functional.l1_loss(alignment_fine, alignment_coarse)

        if iteration < tacotron_params['ga_iter_limit']:  # FIRST, WE WILL PLAY WITH FIXED GUIDED ATTENTION
            ga_loss = GuidedAttentionLoss(alignment_fine, input_lens, out_lengths)
            ga_loss = ga_loss * tacotron_params['ga_alpha']
        else:
            ga_loss = 0.0

        return mel_loss + gate_loss + decoder_coarse_loss + attention_loss + ga_loss
