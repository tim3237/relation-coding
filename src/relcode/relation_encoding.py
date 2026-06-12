import math
from typing import Optional, Tuple, Union, List

import torch

try:
    # Try to import transformer Cache.
    from transformers.cache_utils import Cache
except ModuleNotFoundError as _:
    # Fallback for environments where transformers is not installed
    class Cache:
        """Cache dummy class

        Raises:
            EnvironmentError: Always. This class is not meant to be instantiated.
        """
        def __init__(self, *args, **kwargs):
            raise EnvironmentError("This is just a dummy class. Install transformers to use cache.")

        def update(self, *args, **kwargs):
            raise EnvironmentError("This is just a dummy class. Install transformers to use cache.")

        def get_seq_length(self, *args, **kwargs):
            raise EnvironmentError("This is just a dummy class. Install transformers to use cache.")


def position_to_relation_labels(position_ids: torch.Tensor,
                                 is_causal: bool,
                                 cap_dist: int,
                                 cache_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Translates sequence of position IDs into a square-matrix relation label tensor.

    Args:
        position_ids: Positions of new inputs. Shape: [..., in_length]
        is_causal: If True, future positions are simply labelled as distance -1, the rightmost relative position class.
        cap_dist: Any position beyond cap_dist is capped to cap_dist.
        cache_positions: Positions present in cache -- they cannot attend to any new tokens but should be present as
            past context. Shape: [..., cache_length]

    Returns:
        Square-matrix tensor of binary relation labels. Shape: [..., in_length, in_length + cache_length]
    """
    if cache_positions is None:
        distances = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
    else:
        distances = torch.cat([cache_positions, position_ids], dim=-1).unsqueeze(-2) - position_ids.unsqueeze(-1)
    if is_causal:
        distances = distances.clamp(min=-cap_dist, max=-1)
    else:
        distances = distances.clamp(min=-cap_dist, max=cap_dist)
    # add one to negative, resulting position 0 to be labelled as distance -1. Might be ignored anyway but must be valid
    return (distances + 0.5).long() + cap_dist - 1


class RelationEncoder(torch.nn.Module):
    """Incorporates binary relations into a sequence of embeddings for identification using dot-product attention.

    Note:
        Relations are given as square-matrices of relation identifiers, i.e., this implementation only supports a
        single relation for each 2-tuple of input sequence positions.

    Args:
        hidden_size: Dimensionality of the input embeddings
        num_relations: Number of relation classes to encode (commonly num_attention_heads or num_key_value_heads)
        relation_dim: Number of dimensions for each relation head (commonly attention_head_size or head_dim)
        mix: Mixing approach for value-only and attention iterations:
            "full": Use both positive and negative influence guided by relation labels.
            "positive": Use only positive influence guided by relation labels.
        num_iter: Number of mixing iterations. The first value is value-only iteration and the second specifies the
            number of attention-based mixing.
        residual_weight: Float in range [0.0, 1.0), specifies the weight of the residual connection applied to the
            emphasis iteration output which will be weighted 1.0-residual_weight. Can be specified for each iteration
            individually.
        attention_bias: Parameter for linear layers
        layer_ids: Used to address past_key_values in caches. Defaults to [0, ..., num_iter-1]

    Raises:
        ValueError: If residual_weight is given as tuple of invalid length
    """
    def __init__(
            self,
            hidden_size: int,
            num_relations: int,
            relation_dim: Optional[int] = None,
            mix: Tuple[str, str] = ("full", "full"),
            num_iter: Tuple[int, int] = (2, 1),
            residual_weight: Optional[Union[Tuple, float]] = 0.5,
            attention_bias: bool = True,
            layer_ids: Optional[Union[List, Tuple]] = None
    ):
        super().__init__()
        self.num_relations = num_relations
        self.relation_dim = max(hidden_size // self.num_relations, 8) if relation_dim is None else relation_dim

        self.value_mix, self.attention_mix = mix
        self.num_value_iter, self.num_attention_iter = num_iter

        if isinstance(residual_weight, float):
            self.residual_weights = (residual_weight,)*sum(num_iter)
        else:
            if len(residual_weight) != sum(num_iter):
                raise ValueError("The number of specified residual weights must match the total number of iterations.")
            self.residual_weights = residual_weight

        self.all_enc_embed_size = self.num_relations * self.relation_dim
        if self.num_attention_iter > 0:
            self.softmax = torch.nn.Softmax(dim=-1)
            self.query = torch.nn.Linear(hidden_size, self.all_enc_embed_size, bias=attention_bias)
            self.key = torch.nn.Linear(hidden_size, self.all_enc_embed_size, bias=attention_bias)
        self.value = torch.nn.Linear(hidden_size, self.all_enc_embed_size, bias=attention_bias)
        self.out_projection = torch.nn.Linear(self.all_enc_embed_size, hidden_size, bias=attention_bias)
        self.layer_ids = layer_ids

    def _split_heads(self, embeddings):
        return embeddings.view(
            embeddings.shape[:-1] + (self.num_relations, self.relation_dim)
        ).permute(0, 2, 1, 3)

    def _layer_id(self, lid):
        """Translates iteration running index i to cache layer ID.
        """
        lid = lid if self.layer_ids is None else self.layer_ids[lid]
        return lid

    def forward(self,
                embeddings: torch.Tensor,
                relation_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Cache] = None) -> torch.Tensor:
        """Encodes binary relations.

        Args:
            embeddings: Input host embeddings for relations. Shape: [batch_size, in_len, hidden_size]
            relation_ids: Square-matrix tensor of binary relation labels. Shape: [batch_size, in_len, seq_len]
            attention_mask: 0-1 attention mask
            past_key_value: Transformers style cache for efficient inference

        Returns:
            Host embeddings with emphasised relations. Shape: [batch_size, in_len, hidden_size]
        """
        if attention_mask is None:
            attention_mask = 1
        else:
            attention_mask = attention_mask.unsqueeze(1)  # broadcastable head dimension

        # Prepare values which are constant for each emphasis iteration.
        batch_size, in_len, seq_len, = relation_ids.shape
        selection_mask = torch.zeros((batch_size, self.num_relations, in_len, seq_len), device=embeddings.device)
        selection_mask = selection_mask.scatter(1, relation_ids.unsqueeze(1), 1.0)
        if self.num_value_iter > 0 and self.value_mix != "positive":
            avoid_mask = (1 - selection_mask) * attention_mask  # Avoidance will translate to negative context response.
        else:
            avoid_mask = None
        selection_mask = selection_mask * attention_mask
        relation_selection_count = selection_mask.sum(-1, keepdim=True)
        target_response_probs = selection_mask * (1.0/relation_selection_count.clamp(min=1))

        if self.num_value_iter > 0:
            if self.value_mix == "positive":
                correction_scores = target_response_probs
            else:
                avoid_count = avoid_mask.sum(-1, keepdim=True)
                correction_scores = (
                    selection_mask * (2.0/relation_selection_count.clamp(min=1))
                    - avoid_mask * (1.0/avoid_count.clamp(min=1))
                )
        else:
            correction_scores = None

        if self.num_attention_iter > 0:
            add_attention_mask = (1.0 - attention_mask) * torch.finfo(embeddings.dtype).min
        else:
            add_attention_mask = None

        for i in range(self.num_value_iter + self.num_attention_iter):
            values = self._split_heads(self.value(embeddings))
            if i < self.num_value_iter:
                keys = torch.zeros_like(values)
            else:
                keys = self._split_heads(self.key(embeddings))
            if past_key_value is not None:
                keys, values = past_key_value.update(keys, values, self._layer_id(i))
            if i >= self.num_value_iter:
                queries = self._split_heads(self.query(embeddings))
                attention_scores = (torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.relation_dim)
                                    ).clamp(min=-50, max=50)
                attention_response = self.softmax(add_attention_mask - attention_scores) * selection_mask
                if self.attention_mix != "positive":
                    attention_response = attention_response - self.softmax(
                        attention_scores + add_attention_mask) * (1 - selection_mask)
                response_weights = attention_response + target_response_probs
                correction_scores = response_weights / response_weights.abs().sum(dim=-1, keepdim=True).clamp(min=1e-5)

            context_response = torch.matmul(correction_scores, values)
            b, h, n, d_h = context_response.shape
            # shape [b,h,n,d_h] -> [b,n,h,d_h] -> [b,n,d]
            context_response = context_response.permute(0, 2, 1, 3).contiguous().view(
                (b, n, self.all_enc_embed_size))

            relation_embeddings = self.out_projection(context_response)
            res_weight = self.residual_weights[i]
            embeddings = res_weight * embeddings + (1.0 - res_weight) * relation_embeddings

        return embeddings


class RIPE(RelationEncoder):
    """Relational Input Position Encoder

    Args:
        hidden_size: Dimensionality of the input embeddings
        cap_dist: Any position beyond cap_dist is capped to cap_dist (in either direction).
        causal: If true, sets attention_mask default to unidirectional attention and covers left-side distances
            relations only.
        **kwargs: Forwarded to :class:RelationEncoder
    """
    def __init__(self, hidden_size: int, cap_dist: int = 4, causal: bool = False, **kwargs):
        self.causal = causal
        self.cap_dist = cap_dist
        if self.causal:
            num_relations = cap_dist  # -cap_dist,...-1
        else:
            num_relations = (2 * cap_dist)  # -cap_dist,...-1, 1,...,cap_dist
        super().__init__(
            hidden_size, num_relations, **kwargs)

    def forward(self,
                embeddings: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Cache] = None) -> torch.Tensor:
        """Encodes binary relations derived from position_ids.

        Note:
            For cache usage, we assume positions_ids to be consecutive in order to work with cache implementations that
            do not keep track of token positions.

        Args:
            embeddings: Input host embeddings for relations. Shape: [batch_size, in_len, hidden_size]
            position_ids: Sequence positions of input embeddings. Shape: [batch_size, in_len]
            attention_mask: 0-1 attention mask. Defaults to causal attention mask if self.causal
            past_key_value: Transformers style cache for efficient inference

        Returns:
            Host embeddings with emphasised position relations. Shape: [batch_size, in_len, hidden_size]

        Raises:
            ValueError: If position_ids lack the sequence dimension
        """
        if past_key_value is not None:
            if len(position_ids.shape) != 2:
                raise ValueError("position_ids are expected to be of shape [batch_size, new_seq_len]. " +
                                 f"Provided shape: {position_ids.shape}")
            cache_len = past_key_value.get_seq_length(
                0 if self.layer_ids is None else self.layer_ids[0], position_ids.shape[-1]) - position_ids.shape[-1]
            if cache_len > 0:
                min_pos_id = position_ids.min(dim=-1)[0]
                cache_starts = min_pos_id - cache_len
                in_cache_positions = torch.arange(cache_len, device=embeddings.device)
                cache_positions = in_cache_positions.unsqueeze(0) + cache_starts.unsqueeze(1)
            else:
                cache_positions = None
        else:
            cache_len = 0
            cache_positions = None

        all_rel_distance_ids = positions_to_relation_labels(
            position_ids, self.causal, self.cap_dist, cache_positions=cache_positions)
        if self.causal and attention_mask is None:
            # broadcastable 1-0-mask
            attention_mask = torch.ones_like(all_rel_distance_ids[:1]).tril(diagonal=cache_len)

        return super().forward(embeddings, all_rel_distance_ids, attention_mask, past_key_value=past_key_value)
