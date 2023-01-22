import torch

from torch import Tensor


def make_mask_2d(lengths: torch.Tensor, expand_dim: int = None):
    """Create binary mask from lengths indicating which indices are padding"""
    # Make sure `lengths` is a 1d array
    assert len(lengths.shape) == 1

    max_length = torch.amax(lengths, dim=0).item()
    mask = torch.arange(max_length).expand((lengths.shape[0], max_length))  # Shape batch x timesteps
    mask = torch.ge(mask, lengths.unsqueeze(1))

    if expand_dim is not None:
        mask = mask.unsqueeze(2)
        mask = mask.expand((mask.shape[0], mask.shape[1], expand_dim))

    return mask


def make_mask_3d(source_lengths: torch.Tensor, target_lengths: torch.Tensor):
    """
    Make binary mask indicating which combinations of indices involve at least 1 padding element.
    Can be used to mask, for example, a batch attention matrix between 2 sequences
    """
    # Calculate binary masks for source and target
    # Then invert boolean values and convert to float (necessary for bmm later)
    source_mask = (~ make_mask_2d(source_lengths)).float()
    target_mask = (~ make_mask_2d(target_lengths)).float()

    # Add dummy dimensions for bmm
    source_mask = source_mask.unsqueeze(2)
    target_mask = target_mask.unsqueeze(1)

    # Calculate combinations by batch matrix multiplication
    mask = torch.bmm(source_mask, target_mask).bool()
    # Invert boolean values
    mask = torch.logical_not(mask)
    return mask


def _mask_sequences_for_pooling(sequences: Tensor, lengths: Tensor, value: float) -> Tensor:
    lengths = torch.clamp(lengths, min=1)
    mask = make_mask_2d(lengths, expand_dim=sequences.shape[2])
    mask = mask.to(sequences.device)
    sequences = torch.masked_fill(sequences, mask=mask, value=value)
    return sequences


def _max_over_time_pooling(sequences: Tensor, lengths: Tensor) -> Tensor:
    sequences = _mask_sequences_for_pooling(sequences, lengths, value=-torch.inf)
    pooled = torch.max(sequences, dim=1).values

    return pooled


def _sum_over_time_pooling(sequences: Tensor, lengths: Tensor) -> Tensor:
    sequences = _mask_sequences_for_pooling(sequences, lengths, value=0.0)
    pooled = torch.sum(sequences, dim=1)

    return pooled


def _mean_over_time_pooling(sequences: Tensor, lengths: Tensor) -> Tensor:
    sum_pooled = _sum_over_time_pooling(sequences, lengths)
    length_normaliser = lengths.float().reshape(len(lengths), 1).expand(sum_pooled.shape).to(sum_pooled.device)
    pooled = sum_pooled / length_normaliser

    return pooled
