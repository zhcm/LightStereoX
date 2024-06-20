import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionRelative(nn.MultiheadAttention):
    """
    Multihead attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionRelative, self).__init__(embed_dim, num_heads)

    def forward(self, query, key, value, attn_mask=None, pos_enc=None, *args, **kwargs):
        """
        Multihead attention
        :param query: [w, h*bz, C]
        :param key: [w, h*bz, C]
        :param value: [w, h*bz, C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [w, w]
        :param pos_enc: [w, w, C]
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """

        w, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q = q * scaling
        q = q.contiguous().view(w, bsz, self.num_heads, head_dim)
        k = k.contiguous().view(w, bsz, self.num_heads, head_dim)
        v = v.contiguous().view(w, bsz, self.num_heads, head_dim)  # [w, bsz, num_heads, head_dim]

        if pos_enc is not None:
            weight = self.in_proj_weight[0:2 * embed_dim, :]
            bias = self.in_proj_bias[0:2 * embed_dim]
            q_r, k_r = F.linear(pos_enc, weight, bias).chunk(2, dim=-1)  # [w, w, C], [w, w, C]
            q_r = q_r * scaling
            q_r = q_r.contiguous().view(w, w, self.num_heads, head_dim)
            k_r = k_r.contiguous().view(w, w, self.num_heads, head_dim)
        else:
            q_r = None
            k_r = None

        # compute attn weight
        attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # [bsz, num_heads, w, w]

        # add positional terms
        if pos_enc is not None:
            attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # [bsz, num_heads, w, w]
            attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # [bsz, num_heads, w, w]
            attn_feat = attn_feat + attn_feat_pos + attn_pos_feat

        assert list(attn_feat.size()) == [bsz, self.num_heads, w, w]

        # apply attn mask
        if attn_mask is not None:
            attn_mask = attn_mask[None, None, ...]
            attn_feat += attn_mask

        # raw attn
        raw_attn = attn_feat  # [bsz, num_heads, w, w]

        # softmax
        attn = F.softmax(attn_feat, dim=-1)  # [bsz, num_heads, w, w]
        # [w, bsz, num_heads, head_dim] -> [bsz, num_heads, w, head_dim] -> [bsz*num_heads, w, head_dim]
        v = v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim)

        output = torch.bmm(attn.view(bsz * self.num_heads, w, w), v)  # [bsz*num_heads, w, head_dim]
        assert list(output.size()) == [bsz * self.num_heads, w, head_dim]

        output = output.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim)
        output = F.linear(output, self.out_proj.weight, self.out_proj.bias)

        # average attention weights over heads
        attn = attn.sum(dim=1) / self.num_heads  # [bsz, w, w]

        # sum raw attn weights over heads
        raw_attn = raw_attn.sum(dim=1)  # [bsz, w, w]

        return output, attn, raw_attn
