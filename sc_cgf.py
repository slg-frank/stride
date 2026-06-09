
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalGate(nn.Module):


    def __init__(self, d_model=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(d_model, d_model)

    def forward(self, z_u, z_v):

        gate_weight = self.gate(torch.cat([z_u, z_v], dim=-1))
        v_info = self.transform(z_v)
        return z_u + gate_weight * v_info  # 残差连接


class StructureFusionGate(nn.Module):


    def __init__(self):
        super().__init__()
        # 轻量级MLP
        self.mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
        # 可学习的divergence权重
        self.alpha_div = nn.Parameter(torch.tensor(0.6))

    def compute_divergence(self, z_u, z_v, pred_u, pred_v):


        feat_div = 1 - F.cosine_similarity(z_u, z_v, dim=-1)
        feat_div = torch.clamp(feat_div, min=1e-8, max=1.0)  # 避免数值问题


        pred_div = torch.abs(pred_u - pred_v)
        pred_div = torch.clamp(pred_div, min=1e-8, max=1.0)


        alpha = torch.sigmoid(self.alpha_div)
        return alpha * feat_div + (1 - alpha) * pred_div

    def forward(self, tcp_m, tcp_t, tcp_l, z_m, z_t, z_l, pred_m, pred_t, pred_l):


        d_mt = self.compute_divergence(z_m, z_t, pred_m, pred_t)
        d_ml = self.compute_divergence(z_m, z_l, pred_m, pred_l)
        d_tl = self.compute_divergence(z_t, z_l, pred_t, pred_l)


        gate_input = torch.stack([tcp_m, tcp_t, tcp_l, d_mt, d_ml, d_tl], dim=-1)

        alpha_logits = self.mlp(gate_input)
        alpha_weights = F.softmax(alpha_logits, dim=-1)

        alpha = {
            'self': alpha_weights[:, 0],
            'mt': alpha_weights[:, 1],
            'ml': alpha_weights[:, 2],
            'tl': alpha_weights[:, 3]
        }

        divergences = {'d_mt': d_mt, 'd_ml': d_ml, 'd_tl': d_tl}

        return alpha, divergences


class SC_CGF_FusionLayer(nn.Module):


    def __init__(self, d_model=64):
        super().__init__()


        self.cross_gate_mt = CrossModalGate(d_model)
        self.cross_gate_ml = CrossModalGate(d_model)
        self.cross_gate_tl = CrossModalGate(d_model)


        self.sfg = StructureFusionGate()


        self.clf_m = nn.Linear(d_model, 2)
        self.clf_t = nn.Linear(d_model, 2)
        self.clf_l = nn.Linear(d_model, 2)

    def forward(self, new_metric, new_trace, new_log,
                tcp_metric, tcp_trace, tcp_log):


        h_self = (tcp_metric.unsqueeze(-1) * new_metric +
                  tcp_trace.unsqueeze(-1) * new_trace +
                  tcp_log.unsqueeze(-1) * new_log)


        h_mt = self.cross_gate_mt(new_metric, new_trace)
        h_ml = self.cross_gate_ml(new_metric, new_log)
        h_tl = self.cross_gate_tl(new_trace, new_log)


        pred_m = F.softmax(self.clf_m(new_metric), dim=-1)[:, 1]
        pred_t = F.softmax(self.clf_t(new_trace), dim=-1)[:, 1]
        pred_l = F.softmax(self.clf_l(new_log), dim=-1)[:, 1]


        alpha, divergences = self.sfg(
            tcp_metric, tcp_trace, tcp_log,
            new_metric, new_trace, new_log,
            pred_m, pred_t, pred_l
        )


        h_fused = (alpha['self'].unsqueeze(-1) * h_self +
                   alpha['mt'].unsqueeze(-1) * h_mt +
                   alpha['ml'].unsqueeze(-1) * h_ml +
                   alpha['tl'].unsqueeze(-1) * h_tl)


        aux_outputs = {
            'alpha': alpha,
            'divergences': divergences,
            'pred_m': pred_m,
            'pred_t': pred_t,
            'pred_l': pred_l,
            'h_self': h_self,
            'h_mt': h_mt,
            'h_ml': h_ml,
            'h_tl': h_tl
        }

        return h_fused, aux_outputs


def compute_sccgf_loss(aux_outputs, tcp_scores):

    if aux_outputs is None:
        return torch.tensor(0.0)

    alpha = aux_outputs['alpha']
    div = aux_outputs['divergences']


    tcp_mean = (tcp_scores['m'] + tcp_scores['t'] + tcp_scores['l']) / 3
    conf_loss = F.mse_loss(alpha['self'], tcp_mean.detach())


    div_penalty = (
            alpha['mt'] * div['d_mt'] +
            alpha['ml'] * div['d_ml'] +
            alpha['tl'] * div['d_tl']
    ).mean()


    alpha_tensor = torch.stack([
        alpha['self'],
        alpha['mt'],
        alpha['ml'],
        alpha['tl']
    ], dim=-1)
    entropy = -(alpha_tensor * torch.log(alpha_tensor + 1e-8)).sum(dim=-1).mean()


    L_struct = conf_loss + 0.5 * div_penalty - 0.1 * entropy

    return L_struct



if __name__ == '__main__':
    print("=" * 80)
    print("SC-CGF Module Test")
    print("=" * 80)

    # 模拟输入
    batch_size = 32
    d_model = 64

    # 模态特征
    new_metric = torch.randn(batch_size, d_model)
    new_trace = torch.randn(batch_size, d_model)
    new_log = torch.randn(batch_size, d_model)

    # TCP置信度
    tcp_metric = torch.rand(batch_size)
    tcp_trace = torch.rand(batch_size)
    tcp_log = torch.rand(batch_size)

    # 初始化SC-CGF
    sc_cgf = SC_CGF_FusionLayer(d_model=d_model)

    # 前向传播
    h_fused, aux_outputs = sc_cgf(
        new_metric, new_trace, new_log,
        tcp_metric, tcp_trace, tcp_log
    )

    print(f"\n✅ Forward pass successful!")
    print(f"   Fused feature shape: {h_fused.shape}")

    # 打印结构权重
    print(f"\n📊 Structure weights (sample 0):")
    for key, val in aux_outputs['alpha'].items():
        print(f"   α_{key}: {val[0].item():.4f}")

    # 计算loss
    tcp_scores = {'m': tcp_metric, 't': tcp_trace, 'l': tcp_log}
    L_struct = compute_sccgf_loss(aux_outputs, tcp_scores)

    print(f"\n📉 Structure loss: {L_struct.item():.6f}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)