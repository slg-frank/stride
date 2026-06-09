
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# 导入模块
from imad_v2_encoders import MetricEncoder, TraceEncoder, DistilledLogEncoder
from imad_v2_rca import MultiViewRCA
from sc_cgf import SC_CGF_FusionLayer, compute_sccgf_loss




class UncertainBlock(nn.Module):


    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # 编码器（hidden固定128）
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )


        self.fc_mu = nn.Sequential(nn.Linear(128, d_out), nn.Sigmoid())
        self.fc_var = nn.Sequential(nn.Linear(128, d_out), nn.Sigmoid())

        # 注意力机制
        self.attention_mu = SimpleAttention(d_out)
        self.attention_var = SimpleAttention(d_out)

    def reparametrize(self, mu, var):
        """重参数化技巧（用户版本）"""
        std = var.sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, var):

        return -0.5 * torch.mean(torch.sum(1 + var - mu ** 2 - var.exp(), dim=-1))

    def forward(self, encoded):

        # 编码
        encoded = self.encoder(encoded)

        # 均值和方差
        mu = self.fc_mu(encoded)
        var = self.fc_var(encoded)

        # 应用注意力
        mu_attn = self.attention_mu(mu)
        var_attn = self.attention_var(var)

        # 重参数化采样
        new_sample = self.reparametrize(mu_attn, var_attn)
        kl_loss = self.kl_loss(mu_attn, var_attn)

        return new_sample, kl_loss


class SimpleAttention(nn.Module):
    """简单注意力机制（MUAD使用）"""

    def __init__(self, feature_dim):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self._glorot_init()

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights) + self.bias
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights * x

    def _glorot_init(self):
        nn.init.xavier_uniform_(self.attention_weights)


class ConfidenceLearning(nn.Module):
    """置信度学习模块（MUAD核心）"""

    def __init__(self, feature_dim, num_class=2):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_class)
        self.confidence_layer = nn.Linear(feature_dim, 1)

    def forward(self, x, y_true=None, criterion=None):

        logits = self.classifier(x)
        confidence = torch.sigmoid(self.confidence_layer(x)).squeeze()

        if y_true is not None and criterion is not None:
            cls_loss = criterion(logits, y_true)

            prob = F.softmax(logits, dim=1)
            p_target = torch.gather(prob, 1, y_true.unsqueeze(1)).squeeze()
            conf_loss = F.mse_loss(confidence, p_target)

            loss = cls_loss + conf_loss
        else:
            loss = 0

        return logits, confidence, loss




class IMAD_V2(nn.Module):


    def __init__(self, device, event_num, metric_num=7, node_num=12,
                 num_class=2, hidden_dim=64,
                 # MUAD参数
                 kl_weight=0.4, conf_weight=0.01,
                 # IMAD_V2参数
                 llm_dim=4096, student_dim=128,
                 teacher_embeddings=None,
                 use_sccgf=True,
                 use_imad_v2=True,
                 use_multiview_rca=True,
                 # 损失权重
                 alpha_task=0.5,  # 检测vs定位权重
                 lambda_struct=0.1,  # SC-CGF结构损失权重
                 lambda_distill=0.3,  # 蒸馏损失权重
                 # 网络结构
                 graph_hiddens=[64],
                 attn_head=4,
                 attn_drop=0.0,
                 mm_classifier_hiddens=[64]):

        super().__init__()
        self.device = device
        self.node_num = node_num
        self.num_class = num_class
        self.hidden_dim = hidden_dim

        # 损失权重
        self.kl_weight = kl_weight
        self.conf_weight = conf_weight
        self.alpha_task = alpha_task
        self.lambda_struct = lambda_struct
        self.lambda_distill = lambda_distill

        # 功能开关
        self.use_sccgf = use_sccgf
        self.use_imad_v2 = use_imad_v2
        self.use_multiview_rca = use_multiview_rca



        # ========== 编码器 ==========
        self.metric_encoder = MetricEncoder(
            metric_num, hidden_dim, graph_hiddens, device, attn_head, attn_drop
        )
        self.trace_encoder = TraceEncoder(
            hidden_dim, graph_hiddens, device, attn_head, attn_drop
        )

        # Log编码器: 根据配置选择
        if use_imad_v2:
            self.log_encoder = DistilledLogEncoder(
                event_num, hidden_dim, graph_hiddens, device,
                llm_dim=llm_dim,
                student_dim=student_dim,
                teacher_embeddings=teacher_embeddings,
                attn_head=attn_head,
                attn_drop=attn_drop,
                use_distillation=True
            )
        else:
            # 降级到简单的Linear encoder
            from imad_v2_encoders import GraphModel
            self.log_encoder = nn.ModuleDict({
                'log_model': nn.Linear(event_num, hidden_dim),
                'graph_model': GraphModel(hidden_dim, graph_hiddens, device, attn_head, attn_drop)
            })


        # 多模态分类器（融合后）
        if len(mm_classifier_hiddens) > 0:
            layers = []
            in_dim = hidden_dim
            for h_dim in mm_classifier_hiddens:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                in_dim = h_dim
            self.mm_classifier = nn.Sequential(*layers)
            final_dim = mm_classifier_hiddens[-1]
        else:
            self.mm_classifier = nn.Identity()
            final_dim = hidden_dim


        # ========== 定位分支 ==========
        if use_multiview_rca:
            # 改进3: Multi-View RCA
            self.localizer = MultiViewRCA(hidden_dim, node_num)
        else:
            # 降级到简单的MLP
            self.localizer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, node_num)
            )

    def forward(self, graph, culprit_labels=None, return_aux=False):

        batch_size = graph.batch_size

        # ========== Step 1: 编码 ==========
        metric_out = self.metric_encoder(graph)  # [batch, hidden_dim]
        trace_out = self.trace_encoder(graph)

        # Log编码（可能返回蒸馏损失）
        if self.use_imad_v2:
            log_out, distill_loss = self.log_encoder(graph, return_distill_loss=True)
        else:
            x = graph.ndata['logs']
            x = self.log_encoder['log_model'](x)
            log_out = self.log_encoder['graph_model'](graph, x)
            distill_loss = torch.tensor(0.0, device=self.device)

        # ========== Step 2: 不确定性估计 ==========
        new_metric, kl_metric = self.uncertain_metric(metric_out)
        new_trace, kl_trace = self.uncertain_trace(trace_out)
        new_log, kl_log = self.uncertain_log(log_out)

        # KL loss (标量，用于MUAD损失)
        kl_loss = kl_metric + kl_trace + kl_log

        # TCP置信度（通过学习层计算，MUAD原版方式）
        # shape: [batch] - 通过Linear层 + Sigmoid学习
        tcp_metric = torch.sigmoid(self.TCPConfidenceLayer_metric(new_metric)).squeeze()
        tcp_trace = torch.sigmoid(self.TCPConfidenceLayer_trace(new_trace)).squeeze()
        tcp_log = torch.sigmoid(self.TCPConfidenceLayer_log(new_log)).squeeze()

        # ========== Step 3: 融合 ==========
        if self.use_sccgf:
            # 改进1: SC-CGF动态融合
            h_fused, aux_outputs = self.fusion_layer(
                new_metric, new_trace, new_log,
                tcp_metric, tcp_trace, tcp_log
            )

            # SC-CGF结构损失
            tcp_scores = {'m': tcp_metric, 't': tcp_trace, 'l': tcp_log}
            struct_loss = compute_sccgf_loss(aux_outputs, tcp_scores)
        else:
            # 简单加权融合
            h_fused = (
                    tcp_metric.unsqueeze(-1) * new_metric +
                    tcp_trace.unsqueeze(-1) * new_trace +
                    tcp_log.unsqueeze(-1) * new_log
            )
            aux_outputs = None
            struct_loss = torch.tensor(0.0, device=self.device)

        # ========== Step 4: 检测分支 ==========
        if culprit_labels is not None:
            # 构造检测标签（-1→正常0, 其他→异常1）
            detect_labels = (culprit_labels != -1).long()
            criterion = nn.CrossEntropyLoss()



            # Metric
            metric_logits = self.metric_classifier(new_metric)
            metric_prob = F.softmax(metric_logits, dim=1)
            metric_p_target = torch.gather(metric_prob, 1, detect_labels.unsqueeze(1)).squeeze()
            confidence_loss_metric = (
                    F.mse_loss(tcp_metric, metric_p_target) +
                    criterion(metric_logits, detect_labels)
            )

            # Trace
            trace_logits = self.trace_classifier(new_trace)
            trace_prob = F.softmax(trace_logits, dim=1)
            trace_p_target = torch.gather(trace_prob, 1, detect_labels.unsqueeze(1)).squeeze()
            confidence_loss_trace = (
                    F.mse_loss(tcp_trace, trace_p_target) +
                    criterion(trace_logits, detect_labels)
            )

            # Log
            log_logits = self.log_classifier(new_log)
            log_prob = F.softmax(log_logits, dim=1)
            log_p_target = torch.gather(log_prob, 1, detect_labels.unsqueeze(1)).squeeze()
            confidence_loss_log = (
                    F.mse_loss(tcp_log, log_p_target) +
                    criterion(log_logits, detect_labels)
            )

            # 模态级总损失（3个模态）
            confidence_loss = confidence_loss_metric + confidence_loss_trace + confidence_loss_log

            # ========== 融合级分类（主分类器）==========
            h_detect = self.mm_classifier(h_fused)  # [batch, final_dim]

            # MM分类器
            detect_logits = self.fusion_classifier(h_detect)
            MMLoss = criterion(detect_logits, detect_labels)

            # 融合置信度（confidence_loss2）
            detect_conf = torch.sigmoid(self.fusion_confidence(h_detect)).squeeze()
            fusion_prob = F.softmax(detect_logits, dim=1)
            fusion_p_target = torch.gather(fusion_prob, 1, detect_labels.unsqueeze(1)).squeeze()
            confidence_loss2 = F.mse_loss(detect_conf, fusion_p_target)


            modality_conf_weight = 0.6  # MUAD硬编码值
            conf_loss = MMLoss + modality_conf_weight * confidence_loss + confidence_loss2
        else:
            # 推理模式
            h_detect = self.mm_classifier(h_fused)
            detect_logits = self.fusion_classifier(h_detect)
            detect_conf = torch.sigmoid(self.fusion_confidence(h_detect)).squeeze()
            conf_loss = torch.tensor(0.0, device=self.device)


        if self.use_multiview_rca and aux_outputs is not None:
            # 改进3: Multi-View RCA
            if culprit_labels is not None:
                criterion_locate = nn.CrossEntropyLoss(ignore_index=-1)
                locate_loss, locate_loss_dict = self.localizer.compute_multiview_loss(
                    aux_outputs, culprit_labels, criterion_locate
                )
                locate_logits = self.localizer(aux_outputs)
            else:
                locate_logits = self.localizer(aux_outputs)
                locate_loss = torch.tensor(0.0, device=self.device)
                locate_loss_dict = {}
        else:
            # 简单定位
            locate_logits = self.localizer(h_fused)
            if culprit_labels is not None:
                criterion_locate = nn.CrossEntropyLoss(ignore_index=-1)
                locate_loss = criterion_locate(locate_logits, culprit_labels)
                locate_loss_dict = {'simple': locate_loss}
            else:
                locate_loss = torch.tensor(0.0, device=self.device)
                locate_loss_dict = {}


        # detection_loss = kl_loss + conf_loss
        detection_loss = self.kl_weight * kl_loss + conf_loss

        # 总损失（兼容EADRO风格的alpha平衡）
        total_loss = (
                self.alpha_task * detection_loss +
                (1 - self.alpha_task) * locate_loss +
                self.lambda_struct * struct_loss +
                self.lambda_distill * distill_loss
        )

        # 组织输出
        outputs = {
            'detect_logits': detect_logits,
            'detect_conf': detect_conf,
            'locate_logits': locate_logits,
            'total_loss': total_loss,
            'losses': {
                'muad': detection_loss,  # 检测损失（包含kl+conf）
                'kl': kl_loss,
                'conf': conf_loss,  # 已包含MMLoss + 0.6*confidence_loss + confidence_loss2
                'locate': locate_loss,
                'struct': struct_loss,
                'distill': distill_loss,
                'locate_detail': locate_loss_dict
            }
        }

        if return_aux:
            outputs['aux'] = {
                'tcp_scores': tcp_scores if self.use_sccgf else None,
                'aux_outputs': aux_outputs,
                'h_fused': h_fused
            }

        return outputs

