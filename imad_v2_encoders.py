
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling


# ========== 基础组件（复用MUAD） ==========

class GRUEncoder(nn.Module):
    """GRU编码器（用于Metrics和Traces）"""

    def __init__(self, in_size, out_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=out_dim,
            batch_first=True,
            bidirectional=False
        )
        self.out_dim = out_dim

    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden.squeeze(0)


class GraphModel(nn.Module):
    """图注意力网络（用于所有编码器）"""

    def __init__(self, in_dim, linear_sizes, device, attn_head=4, attn_drop=0.0):
        super().__init__()
        self.device = device

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for hidden in linear_sizes:
            self.gat_layers.append(
                GATv2Conv(
                    in_feats=in_dim if len(self.gat_layers) == 0 else linear_sizes[len(self.gat_layers) - 1],
                    out_feats=hidden,
                    num_heads=attn_head,
                    feat_drop=attn_drop,
                    attn_drop=attn_drop,
                    residual=True,
                    allow_zero_in_degree=True
                )
            )

        # Global pooling
        gate_nn = nn.Linear(linear_sizes[-1], 1)
        self.pooling = GlobalAttentionPooling(gate_nn)

    def forward(self, graph, x):
        for gat in self.gat_layers:
            x = gat(graph, x)
            x = torch.max(x, dim=1)[0]  # MaxPool over attention heads
            x = F.relu(x)

        # Global pooling
        x = self.pooling(graph, x)
        return x


# ========== IMAD_V2核心：DistilledLogEncoder ==========

class DistilledLogEncoder(nn.Module):


    def __init__(self, event_num, hidden_dim, graph_hiddens, device,
                 llm_dim=4096, student_dim=128,
                 teacher_embeddings=None,
                 attn_head=4, attn_drop=0.0,
                 use_distillation=True):
        """
        Args:
            event_num: 事件类型数量 (如44)
            hidden_dim: 图模型输出维度 (如64)
            graph_hiddens: 图模型隐藏层 (如[64])
            device: 设备
            llm_dim: LLM embedding维度 (如4096)
            student_dim: Student网络中间维度 (如128)
            teacher_embeddings: [event_num, llm_dim] 预训练的LLM embeddings
            use_distillation: 是否启用蒸馏
        """
        super().__init__()
        self.device = device
        self.event_num = event_num
        self.llm_dim = llm_dim
        self.student_dim = student_dim
        self.use_distillation = use_distillation

        # ========== Student Network ==========
        # 学习从Hawkes强度(event_num维) → student_dim → hidden_dim
        self.student_mlp = nn.Sequential(
            nn.Linear(event_num, student_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(student_dim, hidden_dim)
        )

        # ========== Teacher Embeddings (固定) ==========
        if teacher_embeddings is not None:
            # 注册为buffer（不参与训练）
            self.register_buffer('teacher_embeddings', teacher_embeddings)

            # Teacher的降维投影（可训练，用于对齐维度）
            self.teacher_proj = nn.Sequential(
                nn.Linear(llm_dim, student_dim),
                nn.ReLU(),
                nn.Linear(student_dim, hidden_dim)
            )
        else:
            self.teacher_embeddings = None
            self.teacher_proj = None

        # ========== 图注意力网络 ==========
        self.graph_model = GraphModel(
            hidden_dim, graph_hiddens, device, attn_head, attn_drop
        )

        # ========== 置信度学习（可选） ==========
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, graph, return_distill_loss=False):
        """
        Args:
            graph: DGL graph，包含 graph.ndata['logs'] [batch*node, event_num]
            return_distill_loss: 是否返回蒸馏损失

        Returns:
            encoded: [batch, hidden_dim] 编码后的特征
            distill_loss: 蒸馏损失（如果return_distill_loss=True）
        """
        logs = graph.ndata['logs']  # [batch*node, event_num]

        # ========== Student Forward ==========
        student_out = self.student_mlp(logs)  # [batch*node, hidden_dim]

        # ========== Teacher Forward (如果启用蒸馏) ==========
        distill_loss = torch.tensor(0.0, device=self.device)

        if self.use_distillation and self.teacher_embeddings is not None and return_distill_loss:
            # Teacher: 加权组合LLM embeddings
            # logs: [batch*node, event_num]
            # teacher_embeddings: [event_num, llm_dim]
            # → teacher_weighted: [batch*node, llm_dim]

            # 归一化Hawkes强度作为权重
            log_weights = F.softmax(logs, dim=-1)  # [batch*node, event_num]
            teacher_weighted = torch.matmul(log_weights, self.teacher_embeddings)  # [batch*node, llm_dim]

            # Teacher投影到hidden_dim
            teacher_out = self.teacher_proj(teacher_weighted)  # [batch*node, hidden_dim]

            # 蒸馏损失: KL散度
            distill_loss = self._compute_distillation_loss(student_out, teacher_out)

        # ========== 图注意力 ==========
        encoded = self.graph_model(graph, student_out)  # [batch, hidden_dim]

        if return_distill_loss:
            return encoded, distill_loss
        else:
            return encoded

    def _compute_distillation_loss(self, student_out, teacher_out):
        """
        计算蒸馏损失

        使用MSE损失（也可以用KL散度）
        """
        # L2归一化
        student_norm = F.normalize(student_out, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_out, p=2, dim=-1)

        # MSE loss
        mse_loss = F.mse_loss(student_norm, teacher_norm)

        # 可选: 余弦相似度损失
        cos_sim = F.cosine_similarity(student_out, teacher_out, dim=-1)
        cos_loss = (1 - cos_sim).mean()

        # 组合损失
        distill_loss = 0.7 * mse_loss + 0.3 * cos_loss

        return distill_loss

    def get_confidence(self, encoded):
        """获取置信度分数"""
        return self.confidence_layer(encoded).squeeze(-1)


# ========== Metric和Trace编码器（保持不变） ==========

class MetricEncoder(nn.Module):
    """Metrics编码器（与MUAD相同）"""

    def __init__(self, metric_num, hidden_dim, graph_hiddens,
                 device, attn_head=4, attn_drop=0.0):
        super().__init__()
        self.metric_model = GRUEncoder(metric_num, hidden_dim)
        self.graph_model = GraphModel(
            hidden_dim, graph_hiddens, device, attn_head, attn_drop
        )

    def forward(self, graph):
        x = graph.ndata['metrics']
        x = self.metric_model(x)
        x = self.graph_model(graph, x)
        return x


class TraceEncoder(nn.Module):
    """Traces编码器（与MUAD相同）"""

    def __init__(self, hidden_dim, graph_hiddens, device,
                 attn_head=4, attn_drop=0.0):
        super().__init__()
        self.trace_model = GRUEncoder(1, hidden_dim)
        self.graph_model = GraphModel(
            hidden_dim, graph_hiddens, device, attn_head, attn_drop
        )

    def forward(self, graph):
        x = graph.ndata['traces']
        x = self.trace_model(x)
        x = self.graph_model(graph, x)
        return x


# ========== 测试代码 ==========

if __name__ == '__main__':
    print("=" * 80)
    print("IMAD_V2 Encoders Test")
    print("=" * 80)

    import dgl

    # 模拟数据
    batch_size = 32
    node_num = 12
    event_num = 44
    llm_dim = 4096
    hidden_dim = 64

    # 创建图
    edges = ([0, 1, 2], [1, 2, 0])  # 简单的环形图
    g = dgl.graph(edges, num_nodes=node_num)
    g = dgl.batch([g] * batch_size)

    # 模拟特征
    g.ndata['metrics'] = torch.randn(batch_size * node_num, 10, 7)
    g.ndata['traces'] = torch.randn(batch_size * node_num, 10, 1)
    g.ndata['logs'] = torch.rand(batch_size * node_num, event_num)  # Hawkes强度

    # 模拟LLM embeddings
    teacher_embeddings = torch.randn(event_num, llm_dim)

    # 创建编码器
    device = torch.device('cpu')

    print("\n1. Testing MetricEncoder...")
    metric_encoder = MetricEncoder(7, hidden_dim, [64], device)
    metric_out = metric_encoder(g)
    print(f"   ✓ Metric output shape: {metric_out.shape}")

    print("\n2. Testing TraceEncoder...")
    trace_encoder = TraceEncoder(hidden_dim, [64], device)
    trace_out = trace_encoder(g)
    print(f"   ✓ Trace output shape: {trace_out.shape}")

    print("\n3. Testing DistilledLogEncoder (without distillation)...")
    log_encoder_no_distill = DistilledLogEncoder(
        event_num, hidden_dim, [64], device,
        use_distillation=False
    )
    log_out = log_encoder_no_distill(g)
    print(f"   ✓ Log output shape: {log_out.shape}")

    print("\n4. Testing DistilledLogEncoder (with distillation)...")
    log_encoder_distill = DistilledLogEncoder(
        event_num, hidden_dim, [64], device,
        llm_dim=llm_dim,
        student_dim=128,
        teacher_embeddings=teacher_embeddings,
        use_distillation=True
    )
    log_out, distill_loss = log_encoder_distill(g, return_distill_loss=True)
    print(f"   ✓ Log output shape: {log_out.shape}")
    print(f"   ✓ Distillation loss: {distill_loss.item():.6f}")

    print("\n" + "=" * 80)
    print("✅ All encoder tests passed!")
    print("=" * 80)

