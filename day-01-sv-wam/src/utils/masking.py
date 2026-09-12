"""
Action-centered causal mask -- SV-WAM's core attention-scheduling trick.
"""
import torch


def build_action_centered_causal_mask(
    n_context: int,
    n_action: int,
    n_video: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Builds the additive attention-bias mask (0 = visible, -inf = blocked)
    for one joint sequence:

        [ context tokens | action tokens | future-video tokens ]
          <- n_context ->  <- n_action ->  <---- n_video ----->

    Rules (this is SV-WAM's core trick):
      * context  -> attends to context only (encoder-style, bidirectional).
      * action   -> attends to context + other action tokens (causal among
                    themselves), but NEVER to future-video tokens. This is
                    what forces the planner to reason about world dynamics
                    instead of reading the answer off the generated frames.
      * video    -> attends to context + action tokens + earlier video
                    tokens (standard causal video generation).

    Returns:
        mask: [S, S] float tensor, S = n_context + n_action + n_video,
              additive bias ready to pass into
              nn.MultiheadAttention(..., attn_mask=mask).
    """
    s = n_context + n_action + n_video
    mask = torch.zeros(s, s, device=device)

    a0, a1 = n_context, n_context + n_action
    v0, v1 = n_context + n_action, s

    neg_inf = torch.finfo(mask.dtype).min

    # Action tokens: block attention to the entire video block.
    mask[a0:a1, v0:v1] = neg_inf
    # Action tokens are causal among themselves (token i sees tokens <= i).
    action_causal = torch.triu(
        torch.ones(n_action, n_action, device=device), diagonal=1
    ).bool()
    mask[a0:a1, a0:a1] = mask[a0:a1, a0:a1].masked_fill(action_causal, neg_inf)

    # Video tokens: causal among themselves (standard autoregressive/diffusion
    # ordering over time steps), free to see context + all action tokens.
    video_causal = torch.triu(
        torch.ones(n_video, n_video, device=device), diagonal=1
    ).bool()
    mask[v0:v1, v0:v1] = mask[v0:v1, v0:v1].masked_fill(video_causal, neg_inf)

    # Context block stays fully bidirectional (row/col c0:c1 left at 0),
    # matching a standard perception-encoder attention pattern.
    return mask
