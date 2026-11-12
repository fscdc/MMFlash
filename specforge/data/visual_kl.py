# coding=utf-8
"""Per-token visual dependence of a training response, measured on the target model.

MMFlash weights its training loss by how much each response token depends on the image. The dependence of token
t is the KL divergence between the target's next-token distribution with the image and without it, with the text
prompt and the generated prefix fixed:

    d_t = KL( p(. | prefix, image) || p(. | prefix) )

``scripts/score_visual_kl.py`` runs :func:`visual_kl_per_token` over the training corpus and writes the per-token
scores (plus the with-image entropy used as a confidence gate) to a sidecar that training reads through
``data.visual_score_path`` (see :mod:`specforge.data.visual_score`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def visual_kl_per_token(
    target,
    processor,
    record: Dict[str, Any],
    input_ids: torch.Tensor,
    loss_mask_list: Sequence[int],
    max_length: int,
    model_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> Optional[Tuple[Dict[int, float], Dict[int, float]]]:
    """Per token: how much the image moves the target, and how unsure it is.

    Both come from the same pair of forward passes, and the second is what
    separates a real finding from a rediscovery of "uncertain tokens are hard".
    A position where the target has many valid continuations has a distribution
    that moves a lot when the image is dropped AND is hard for any draft, so a
    raw KL-versus-acceptance correlation cannot tell the two apart. The entropy
    of the WITH-image distribution -- the one generation actually samples from
    -- is the control.

    ``model_inputs`` is the processor's full batch for the with-image pass
    (``input_ids`` plus ``pixel_values`` / ``image_grid_thw``). It MUST be
    given for the score to mean what its name says: Qwen3.5's forward only
    scatters image features when ``pixel_values`` is present, so a call with
    bare ``input_ids`` embeds the ``<|image_pad|>`` tokens as ordinary text and
    the "with image" distribution is not conditioned on the picture at all.
    The bare-``input_ids`` fallback is kept only for text-only rows.

    The same conversation is encoded a second time with the image dropped. The
    assistant span is identical text in both, and sits at the end, so the two
    sequences align by a constant offset -- which is asserted token by token
    rather than assumed. ``None`` when they do not align, e.g. because the
    template rendered the image-free turn differently.
    """
    from specforge.data.mm_preprocessing import IMAGE_PLACEHOLDER, to_chat_messages

    if model_inputs is not None:
        if not torch.equal(model_inputs["input_ids"].to(input_ids.device), input_ids):
            raise ValueError(
                "model_inputs['input_ids'] must be the same sequence as input_ids"
            )
    elif record.get("image") is not None:
        raise ValueError(
            "visual_kl_per_token needs the processor batch (pixel_values) for an "
            "image row; pass model_inputs=processor(text=..., images=...)"
        )

    stripped = [
        {**turn, "content": turn["content"].replace(IMAGE_PLACEHOLDER, "").lstrip()}
        for turn in record["conversations"]
    ]
    text = processor.apply_chat_template(
        to_chat_messages(stripped), tokenize=False, add_generation_prompt=False
    )
    blind = processor(text=[text], return_tensors="pt")
    blind_ids = blind["input_ids"].to(input_ids.device)
    if blind_ids.shape[1] > max_length or blind_ids.shape[1] >= input_ids.shape[1]:
        return None

    offset = input_ids.shape[1] - blind_ids.shape[1]
    positions = [index for index, flag in enumerate(loss_mask_list) if flag]
    if not positions:
        return None
    aligned = [index for index in positions if index - offset >= 1]
    if not aligned:
        return None
    if not torch.equal(
        input_ids[0, aligned],
        blind_ids[0, [index - offset for index in aligned]],
    ):
        return None

    # Only the assistant span is scored, and it is contiguous at the end of both
    # sequences, so the trailing window that covers it is all the logits needed.
    # Asking for the whole sequence would materialize (S, 248320) twice -- 700 MB
    # per pass at chartqa's length, and tens of GB on a video prompt.
    seen_len = input_ids.shape[1]
    blind_len = blind_ids.shape[1]
    seen_keep = seen_len - (min(aligned) - 1)
    blind_keep = blind_len - (min(aligned) - offset - 1)
    seen_inputs = (
        {k: v.to(input_ids.device) for k, v in model_inputs.items()}
        if model_inputs is not None
        else {"input_ids": input_ids}
    )
    with torch.no_grad():
        seen = target(**seen_inputs, use_cache=False, logits_to_keep=seen_keep).logits
        unseen = target(
            **{k: v.to(input_ids.device) for k, v in blind.items()},
            use_cache=False,
            logits_to_keep=blind_keep,
        ).logits
    # a kept window holds the LAST k positions, so absolute p sits at p - base
    seen_base = seen_len - seen.shape[1]
    blind_base = blind_len - unseen.shape[1]

    kl: Dict[int, float] = {}
    entropy: Dict[int, float] = {}
    # the distribution that produced token t lives at position t - 1
    for start in range(0, len(aligned), 64):
        chunk = aligned[start : start + 64]
        seen_rows = [index - 1 - seen_base for index in chunk]
        blind_rows = [index - offset - 1 - blind_base for index in chunk]
        if min(seen_rows) < 0 or min(blind_rows) < 0:
            return None
        rows_seen = seen[0, seen_rows].float()
        rows_unseen = unseen[0, blind_rows].float()
        log_p = torch.log_softmax(rows_seen, dim=-1)
        log_q = torch.log_softmax(rows_unseen, dim=-1)
        probability = log_p.exp()
        values = (probability * (log_p - log_q)).sum(dim=-1)
        # entropy of the with-image distribution, in nats
        surprise = -(probability * log_p).sum(dim=-1)
        for index, value, bits in zip(chunk, values.tolist(), surprise.tolist()):
            kl[index] = value
            entropy[index] = bits
    return kl, entropy


__all__ = ["visual_kl_per_token"]
