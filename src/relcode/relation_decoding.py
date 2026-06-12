from typing import Optional, Union

import torch


class RelationPredictor(torch.nn.Module):
    """Decodes binary relations using dot-product attention.

    Args:
        hidden_size: Dimensionality of the input embeddings
        num_relations: Number of relation classes to predict (commonly num_attention_heads or num_key_value_heads)
        relation_dim: Number of dimensions for each relation head (commonly attention_head_size or head_dim)
        allow_unknown_relation: If True, adds an auxiliary output relation which might be used as uncertainty / "other"
            class.
        relation_weights: Optional class weight for loss computation. Can be preset strings ("bi_positions",
            "causal_positions") or a tensor.
    """
    def __init__(
            self,
            hidden_size: int,
            num_relations: int,
            relation_dim: Optional[int] = None,
            allow_unknown_relation: bool = False,
            relation_weights: Optional[Union[str, torch.Tensor]] = None):
        super().__init__()

        self.embedding_dim = hidden_size
        self.num_relations = num_relations
        if allow_unknown_relation:
            self.num_relation_heads = self.num_relations + 1
            self.allow_unknown_relation = True
        else:
            self.num_relation_heads = self.num_relations
            self.allow_unknown_relation = False

        if relation_dim is not None:
            self.relation_dim = relation_dim
        else:
            self.relation_dim = max(self.embedding_dim // self.num_relation_heads, 8)

        all_dist_embed_size = self.num_relation_heads * self.relation_dim
        if relation_weights is not None:
            if isinstance(relation_weights, torch.Tensor):
                self.relation_class_weights = relation_weights
            elif relation_weights == "bi_positions":  # For bidirectional position relations
                num_dir_relations = num_relations // 2  # Assumes an even total number of relations
                self.relation_class_weights = torch.pow(
                    1.5, -torch.tensor([i - num_dir_relations + 0.5 for i in range(self.num_relations)]).abs().int()
                )
            elif relation_weights == "causal_positions":  # For left-side unidirectional position relations
                self.relation_class_weights = torch.pow(
                    1.5, -torch.tensor([i for i in range(self.num_relations)])
                )
            else:  # assumes "none"
                self.relation_class_weights = None
        else:
            self.relation_class_weights = None
        self.origin = torch.nn.Linear(self.embedding_dim, all_dist_embed_size)
        self.destination = torch.nn.Linear(self.embedding_dim, all_dist_embed_size)

    def _split_heads(self, embeddings):
        return embeddings.view(
            embeddings.shape[:-1] + (self.num_relation_heads, self.relation_dim)
        ).permute(0, 2, 1, 3)

    def forward(self, embeddings):
        """Computes relation scores via dot-product attention.

        Args:
            embeddings: Input embeddings. Shape: [batch_size, seq_len, hidden_size]

        Returns:
            Relation scores. Shape: [batch_size, seq_len, seq_len, out_relations]
        """
        origin = self.origin(embeddings)
        destination = self.destination(embeddings)

        origin = self._split_heads(origin)
        destination = self._split_heads(destination).transpose(-1, -2)

        scores = torch.matmul(origin, destination)
        return scores.permute(0, 2, 3, 1)

    def compute_loss(
            self,
            scores: torch.Tensor,
            relation_labels: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            min_weight: float = 1e-1,
            reflexive: bool = True
    ) -> torch.Tensor:
        """
        Calculates loss from relations scores and labels.

        Note:
            This function assumes that exactly one relation is present in a directed pair of embeddings. I.e., this
            function calculates classification loss given square-matrix labels, comparing scores of different attention
            heads.

        Args:
            scores: Square-matrix tensor of predicted relation scores. Shape: [batch_size, seq_len, seq_len,
                num_relations]
            relation_labels: Square-matrix tensor, containing labels for possibly reflexive relations. Valid labels are
                non-negative integers. -1 is used as ignore_index. Shape: [batch_size, seq_len, seq_len]
            attention_mask: 0-1 mask.
            min_weight: Relevant if we use unknown_relation for confidence estimation. A positive min_weight ensures
                that a properly optimised model will classify confidently if possible.
                Has no effect if self.allow_unknown_relation=False.
            reflexive: For relative position encoding/decoding, we might want to omit embedding/predicting the
                trivial 0-distance relation. Thus, we can set reflexive=False to ignore labels relating a token with
                itself.
        """
        if not reflexive:
            ignore_mask = 1-torch.eye(relation_labels.shape[-1], device=scores.device).unsqueeze(0)
        else:
            ignore_mask = 1

        if attention_mask is None:
            attention_mask = 1
        elif len(attention_mask.shape) == len(relation_labels.shape) - 1:
            # If attention_mask is 1D, expand it to a 2D relation mask.
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)

        labels = (relation_labels + 1) * ignore_mask * attention_mask - 1

        if self.allow_unknown_relation:
            # The last relation class is interpreted as uncertainty score.
            weight = 1 - torch.nn.functional.softmax(scores, dim=3)[:, :, :, -1] + min_weight
            loss_function = torch.nn.CrossEntropyLoss(
                weight=None if self.relation_class_weights is None else self.relation_class_weights.to(scores.device),
                ignore_index=-1, reduction='none')
            loss = (loss_function(scores.permute(0, 3, 1, 2), labels.long()) * weight).mean()
        else:
            loss_function = torch.nn.CrossEntropyLoss(
                weight=None if self.relation_class_weights is None else self.relation_class_weights.to(scores.device),
                ignore_index=-1)
            loss = loss_function(scores.permute(0, 3, 1, 2), labels.long())
        return loss

