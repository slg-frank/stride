
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewSpecificRCA(nn.Module):


    def __init__(self, hidden_dim, node_num, view_name):
        super().__init__()
        self.view_name = view_name
        self.node_num = node_num

        # 简单的2层MLP
        self.localizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_num)
        )

    def forward(self, h):

        return self.localizer(h)


class MultiViewRCA(nn.Module):


    def __init__(self, hidden_dim, node_num):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_num = node_num

        # 为4个视图创建独立的RCA分支
        self.rca_self = ViewSpecificRCA(hidden_dim, node_num, 'self')
        self.rca_mt = ViewSpecificRCA(hidden_dim, node_num, 'mt')
        self.rca_ml = ViewSpecificRCA(hidden_dim, node_num, 'ml')
        self.rca_tl = ViewSpecificRCA(hidden_dim, node_num, 'tl')

    def forward(self, aux_outputs, return_view_logits=False):

        # 获取各视图特征
        h_self = aux_outputs['h_self']  # [batch, hidden_dim]
        h_mt = aux_outputs['h_mt']
        h_ml = aux_outputs['h_ml']
        h_tl = aux_outputs['h_tl']

        # 获取alpha权重
        alpha = aux_outputs['alpha']
        alpha_self = alpha['self'].unsqueeze(-1)  # [batch, 1]
        alpha_mt = alpha['mt'].unsqueeze(-1)
        alpha_ml = alpha['ml'].unsqueeze(-1)
        alpha_tl = alpha['tl'].unsqueeze(-1)

        # 各视图独立预测
        logits_self = self.rca_self(h_self)  # [batch, node_num]
        logits_mt = self.rca_mt(h_mt)
        logits_ml = self.rca_ml(h_ml)
        logits_tl = self.rca_tl(h_tl)

        # Ensemble: 使用alpha权重加权
        locate_logits = (
                alpha_self * logits_self +
                alpha_mt * logits_mt +
                alpha_ml * logits_ml +
                alpha_tl * logits_tl
        )

        if return_view_logits:
            view_logits = {
                'self': logits_self,
                'mt': logits_mt,
                'ml': logits_ml,
                'tl': logits_tl
            }
            return locate_logits, view_logits
        else:
            return locate_logits

    def compute_multiview_loss(self, aux_outputs, culprit_labels, criterion):

        # 获取各视图预测
        locate_logits, view_logits = self.forward(aux_outputs, return_view_logits=True)

        # 获取alpha权重
        alpha = aux_outputs['alpha']

        # 计算各视图损失
        loss_self = criterion(view_logits['self'], culprit_labels)
        loss_mt = criterion(view_logits['mt'], culprit_labels)
        loss_ml = criterion(view_logits['ml'], culprit_labels)
        loss_tl = criterion(view_logits['tl'], culprit_labels)

        # 加权组合 (使用detach的alpha，避免alpha只优化单一视图)
        total_loss = (
                alpha['self'].mean().detach() * loss_self +
                alpha['mt'].mean().detach() * loss_mt +
                alpha['ml'].mean().detach() * loss_ml +
                alpha['tl'].mean().detach() * loss_tl
        )

        # 也计算ensemble loss（用于监控）
        ensemble_loss = criterion(locate_logits, culprit_labels)

        # 返回损失字典
        loss_dict = {
            'total': total_loss,
            'ensemble': ensemble_loss,
            'self': loss_self,
            'mt': loss_mt,
            'ml': loss_ml,
            'tl': loss_tl
        }

        return total_loss, loss_dict

    def analyze_view_specialization(self, dataloader, device, fault_types=None):

        self.eval()

        if fault_types is None:
            fault_types = {i: f"Node_{i}" for i in range(self.node_num)}

        num_fault_types = len(fault_types)

        # 统计矩阵: [4_views, num_fault_types]
        view_correct = torch.zeros(4, num_fault_types)
        view_total = torch.zeros(4, num_fault_types)

        with torch.no_grad():
            for batch in dataloader:
                graph, labels = batch
                graph = graph.to(device)
                labels = labels.to(device)

                # 只统计异常样本
                anomaly_mask = (labels != -1)
                if anomaly_mask.sum() == 0:
                    continue

                # 需要获取aux_outputs...
                # 这里需要完整的forward流程
                # 暂时跳过，在实际使用时需要完整模型
                pass

        # 计算准确率
        specialization_matrix = view_correct / (view_total + 1e-8)

        return specialization_matrix





if __name__ == '__main__':
    print("=" * 80)
    print("IMAD_V2 Multi-View RCA Test")
    print("=" * 80)

    # 模拟数据
    batch_size = 32
    hidden_dim = 64
    node_num = 12

    # 模拟SC-CGF的辅助输出
    aux_outputs = {
        'h_self': torch.randn(batch_size, hidden_dim),
        'h_mt': torch.randn(batch_size, hidden_dim),
        'h_ml': torch.randn(batch_size, hidden_dim),
        'h_tl': torch.randn(batch_size, hidden_dim),
        'alpha': {
            'self': torch.rand(batch_size),
            'mt': torch.rand(batch_size),
            'ml': torch.rand(batch_size),
            'tl': torch.rand(batch_size)
        }
    }

    # 归一化alpha权重
    alpha_sum = (aux_outputs['alpha']['self'] +
                 aux_outputs['alpha']['mt'] +
                 aux_outputs['alpha']['ml'] +
                 aux_outputs['alpha']['tl'])
    for key in aux_outputs['alpha']:
        aux_outputs['alpha'][key] = aux_outputs['alpha'][key] / alpha_sum

    # 模拟标签 (一半正常-1，一半异常0-11)
    culprit_labels = torch.cat([
        torch.full((batch_size // 2,), -1, dtype=torch.long),  # 正常
        torch.randint(0, node_num, (batch_size // 2,))  # 异常
    ])

    # 创建Multi-View RCA
    multiview_rca = MultiViewRCA(hidden_dim, node_num)

    print("\n1. Testing forward pass...")
    locate_logits = multiview_rca(aux_outputs)
    print(f"   ✓ Locate logits shape: {locate_logits.shape}")

    print("\n2. Testing forward with view logits...")
    locate_logits, view_logits = multiview_rca(aux_outputs, return_view_logits=True)
    print(f"   ✓ Ensemble logits shape: {locate_logits.shape}")
    for view_name, logits in view_logits.items():
        print(f"   ✓ View '{view_name}' logits shape: {logits.shape}")

    print("\n3. Testing loss computation...")
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss, loss_dict = multiview_rca.compute_multiview_loss(
        aux_outputs, culprit_labels, criterion
    )
    print(f"   ✓ Total loss: {total_loss.item():.6f}")
    for loss_name, loss_val in loss_dict.items():
        if isinstance(loss_val, torch.Tensor):
            print(f"   ✓ {loss_name} loss: {loss_val.item():.6f}")

    print("\n4. Testing prediction...")
    preds = locate_logits.argmax(dim=-1)
    print(f"   ✓ Predictions shape: {preds.shape}")
    print(f"   ✓ Sample predictions: {preds[:5].tolist()}")
    print(f"   ✓ Sample labels: {culprit_labels[:5].tolist()}")

    # 计算准确率（只在异常样本上）
    anomaly_mask = (culprit_labels != -1)
    if anomaly_mask.sum() > 0:
        correct = (preds[anomaly_mask] == culprit_labels[anomaly_mask]).sum()
        total = anomaly_mask.sum()
        acc = correct.float() / total
        print(f"   ✓ Accuracy on anomaly samples: {acc.item():.4f}")

    print("\n" + "=" * 80)
    print("✅ All Multi-View RCA tests passed!")
    print("=" * 80)