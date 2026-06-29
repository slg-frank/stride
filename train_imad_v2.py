"""
IMAD_V2 训练脚本

使用方法:
    python train_imad_v2.py --config config_imad_v2.json

作者: IMAD_V2 Team
日期: 2026-01-05
"""

import argparse
import json
import logging
import pickle
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import numpy as np

from imad_v2_model import IMAD_V2


def setup_logging(result_dir):
    """设置日志"""
    os.makedirs(result_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{result_dir}/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class ChunkDataset(Dataset):
    """数据集类"""

    def __init__(self, chunks, chunk_ids, node_num, edges, use_log_embeddings=False):
        self.chunks = chunks
        self.chunk_ids = chunk_ids
        self.node_num = node_num
        self.edges = edges
        self.use_log_embeddings = use_log_embeddings

    def __len__(self):
        return len(self.chunk_ids)

    def __getitem__(self, idx):
        chunk_id = self.chunk_ids[idx]
        chunk = self.chunks[chunk_id]

        # 创建图
        g = dgl.graph(self.edges, num_nodes=self.node_num)

        # 添加节点特征
        g.ndata['metrics'] = torch.FloatTensor(chunk['metrics'])
        g.ndata['traces'] = torch.FloatTensor(chunk['traces'])
        g.ndata['logs'] = torch.FloatTensor(chunk['logs'])

        # 如果有log_embeddings，也添加（用于分析）
        if self.use_log_embeddings and 'log_embeddings' in chunk:
            g.ndata['log_embeddings'] = torch.FloatTensor(chunk['log_embeddings'])

        # 标签
        label = chunk['culprit']

        return g, label


def collate_fn(batch):
    """批处理函数"""
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.LongTensor(labels)
    return batched_graph, labels


def load_data(data_path, metadata_path):
    """
    加载数据并自动验证维度

    自动从实际数据推断metric_num，避免配置错误
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    node_num = metadata['node_num']
    edges = tuple(map(tuple, metadata['edges']))
    event_num = metadata['event_num']
    metric_num_config = metadata.get('metric_num', 7)

    # ===== 自动推断实际维度 =====
    # 从第一个chunk获取实际维度
    first_chunk_id = list(data.keys())[0]
    first_chunk = data[first_chunk_id]

    actual_metric_num = first_chunk['metrics'].shape[-1]
    actual_event_num = first_chunk['logs'].shape[-1]

    # 验证维度
    if metric_num_config != actual_metric_num:
        print(f"\n⚠️  WARNING: metric_num不匹配!")
        print(f"   metadata.json配置: {metric_num_config}")
        print(f"   实际数据维度: {actual_metric_num}")
        print(f"   → 使用实际维度: {actual_metric_num}\n")
        metric_num = actual_metric_num
    else:
        metric_num = metric_num_config

    if event_num != actual_event_num:
        print(f"\n⚠️  WARNING: event_num不匹配!")
        print(f"   metadata.json配置: {event_num}")
        print(f"   实际数据维度: {actual_event_num}")
        print(f"   → 使用实际维度: {actual_event_num}\n")
        event_num = actual_event_num

    return data, node_num, edges, event_num, metric_num


def load_teacher_embeddings(embedding_path):
    """加载LLM teacher embeddings"""
    if not os.path.exists(embedding_path):
        print(f"Warning: Teacher embeddings not found at {embedding_path}")
        return None

    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    print(f"Loaded teacher embeddings: {embeddings.shape}")
    return torch.FloatTensor(embeddings)


def evaluate(model, dataloader, device, phase="Val"):
    """评估模型"""
    model.eval()

    all_detect_preds = []
    all_detect_labels = []
    all_locate_logits = []  # ✅ 添加：存储完整logits用于HR@K计算
    all_locate_labels = []

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            graph, labels = batch
            graph = graph.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(graph, labels)

            # 累计损失
            total_loss += outputs['total_loss'].item()
            num_batches += 1

            # 检测预测
            detect_preds = outputs['detect_logits'].argmax(dim=1)
            detect_labels = (labels != -1).long()
            all_detect_preds.extend(detect_preds.cpu().numpy())
            all_detect_labels.extend(detect_labels.cpu().numpy())

            # ✅ 修改：定位预测 - 收集完整logits而不是argmax
            anomaly_mask = (labels != -1)
            if anomaly_mask.sum() > 0:
                locate_logits_batch = outputs['locate_logits'][anomaly_mask]
                locate_labels_batch = labels[anomaly_mask]
                # 存储完整logits（用于HR@K计算）
                all_locate_logits.extend(locate_logits_batch.cpu().numpy())
                all_locate_labels.extend(locate_labels_batch.cpu().numpy())

    # 计算检测指标
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_detect_labels, all_detect_preds, average='binary', zero_division=0
    )
    detect_acc = accuracy_score(all_detect_labels, all_detect_preds)

    # 计算定位指标（HR@k）
    def hit_rate_at_k(logits_list, labels_list, k):
        """
        计算HR@k - 正确的Top-K实现

        Args:
            logits_list: List of logits [num_samples, node_num]
            labels_list: List of true labels [num_samples]
            k: Top-K
        Returns:
            HR@k: 真实标签在Top-K预测中的比例
        """
        if len(logits_list) == 0:
            return 0.0

        hits = 0
        total = len(labels_list)

        for logits, label in zip(logits_list, labels_list):
            # 获取Top-K预测
            top_k_preds = torch.topk(torch.tensor(logits), k=min(k, len(logits)))[1].numpy()
            # 检查真实标签是否在Top-K中
            if label in top_k_preds:
                hits += 1

        return hits / total if total > 0 else 0.0

    # 计算定位指标（正确的HR@K）
    hr1 = hit_rate_at_k(all_locate_logits, all_locate_labels, 1)
    hr3 = hit_rate_at_k(all_locate_logits, all_locate_labels, 3)
    hr5 = hit_rate_at_k(all_locate_logits, all_locate_labels, 5)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    metrics = {
        'loss': avg_loss,
        'detect_acc': detect_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hr1': hr1,
        'hr3': hr3
    }

    return metrics


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_muad_loss = 0
    total_locate_loss = 0
    total_struct_loss = 0
    total_distill_loss = 0
    num_batches = 0

    for batch in dataloader:
        graph, labels = batch
        graph = graph.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(graph, labels)
        loss = outputs['total_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 累计损失
        total_loss += loss.item()
        total_muad_loss += outputs['losses']['muad'].item()
        total_locate_loss += outputs['losses']['locate'].item()
        total_struct_loss += outputs['losses']['struct'].item()
        total_distill_loss += outputs['losses']['distill'].item()
        num_batches += 1

    avg_losses = {
        'total': total_loss / num_batches,
        'muad': total_muad_loss / num_batches,
        'locate': total_locate_loss / num_batches,
        'struct': total_struct_loss / num_batches,
        'distill': total_distill_loss / num_batches
    }

    return avg_losses


def main(config_path):
    """主函数"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    def get_config(key, default=None):
        """从配置中获取值"""
        if key in config:
            return config[key]
        for sub_config in ['data_config', 'model_config', 'loss_weights',
                           'training_config', 'system_config', 'output_config','feature_switches']:
            if sub_config in config and key in config[sub_config]:
                return config[sub_config][key]
        return default

    # 设置随机种子
    random_seed = get_config('random_seed', 42)
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # 设置日志
    result_dir = get_config('result_dir', './results/imad_v2')
    logger = setup_logging(result_dir)

    logger.info("=" * 70)
    logger.info("IMAD_V2: Integrated Multi-modal Anomaly Detection v2")
    logger.info("=" * 70)

    # 设置设备
    use_gpu = get_config('use_gpu', True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 加载数据
    logger.info("\nLoading data...")
    train_data, node_num, edges, event_num, metric_num = load_data(
        get_config('train_data_path'),
        get_config('metadata_path')
    )
    test_data, _, _, _, _ = load_data(
        get_config('test_data_path'),
        get_config('metadata_path')
    )

    logger.info(f"Data loaded:")
    logger.info(f"  Node num: {node_num}")
    logger.info(f"  Event num: {event_num}")
    logger.info(f"  Metric num: {metric_num}")
    logger.info(f"  Train samples: {len(train_data)}")
    logger.info(f"  Test samples: {len(test_data)}")

    # 加载teacher embeddings
    teacher_embeddings = None
    if get_config('use_imad_v2', True):
        embedding_path = get_config('teacher_embedding_path')
        if embedding_path:
            teacher_embeddings = load_teacher_embeddings(embedding_path)
            if teacher_embeddings is not None:
                teacher_embeddings = teacher_embeddings.to(device)
                logger.info(f"  Teacher embeddings: {teacher_embeddings.shape}")

    # 创建数据集
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)

    # 创建数据加载器
    batch_size = get_config('batch_size', 256)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=get_config('num_workers', 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=get_config('num_workers', 0)
    )

    # 创建模型
    logger.info("\nInitializing IMAD_V2 model...")
    model = IMAD_V2(
        device=device,
        event_num=event_num,
        metric_num=metric_num,
        node_num=node_num,
        num_class=get_config('num_class', 2),
        hidden_dim=get_config('hidden_dim', 64),
        kl_weight=get_config('kl_weight', 0.4),
        conf_weight=get_config('conf_weight', 0.01),
        llm_dim=get_config('llm_dim', 4096),
        student_dim=get_config('student_dim', 128),
        teacher_embeddings=teacher_embeddings,
        use_sccgf=get_config('use_sccgf', True),
        use_imad_v2=get_config('use_imad_v2', True),
        use_multiview_rca=get_config('use_multiview_rca', True),
        alpha_task=get_config('alpha_task', 0.5),
        lambda_struct=get_config('lambda_struct', 0.1),
        lambda_distill=get_config('lambda_distill', 0.3),
        graph_hiddens=get_config('graph_hiddens', [64]),
        attn_head=get_config('attn_head', 4),
        attn_drop=get_config('attn_drop', 0.0),
        mm_classifier_hiddens=get_config('mm_classifier_hiddens', [64])
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=get_config('lr', 0.001))

    # 训练
    logger.info("\nStarting training...")
    best_f1 = 0
    best_hr1 = 0
    patience_counter = 0
    patience = get_config('patience', 15)
    epochs = get_config('epochs', 125)
    eval_every = get_config('eval_every', 5)

    for epoch in range(1, epochs + 1):
        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)

        logger.info(f"\nEpoch {epoch}/{epochs}")
        logger.info(f"  Train Loss: {train_losses['total']:.6f}")
        logger.info(f"    - MUAD: {train_losses['muad']:.6f}")
        logger.info(f"    - Locate: {train_losses['locate']:.6f}")
        logger.info(f"    - Struct: {train_losses['struct']:.6f}")
        logger.info(f"    - Distill: {train_losses['distill']:.6f}")

        # 评估
        if epoch % eval_every == 0:
            test_metrics = evaluate(model, test_loader, device, "Test")

            logger.info(f"  Test Metrics:")
            logger.info(f"    F1: {test_metrics['f1']:.4f}")
            logger.info(f"    Precision: {test_metrics['precision']:.4f}")
            logger.info(f"    Recall: {test_metrics['recall']:.4f}")
            logger.info(f"    HR@1: {test_metrics['hr1']:.4f}")
            logger.info(f"    HR@3: {test_metrics['hr3']:.4f}")

            # 保存最佳模型
            if test_metrics['f1'] > best_f1:
                best_f1 = test_metrics['f1']
                torch.save(model.state_dict(),
                           os.path.join(result_dir, 'best_detect_model.pth'))
                logger.info(f"  → Best F1 updated: {best_f1:.4f}")

            if test_metrics['hr1'] > best_hr1:
                best_hr1 = test_metrics['hr1']
                torch.save(model.state_dict(),
                           os.path.join(result_dir, 'best_rca_model.pth'))
                logger.info(f"  → Best HR@1 updated: {best_hr1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break

    # 最终评估


    model.load_state_dict(torch.load(os.path.join(result_dir, 'best_rca_model.pth')))
    final_metrics = evaluate(model, test_loader, device, "Final")



    # 保存结果
    with open(os.path.join(result_dir, 'final_results.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IMAD_V2 model')
    parser.add_argument('--config', type=str, default='config_imad_v2.json',
                        help='Path to config file')
    args = parser.parse_args()

    main(args.config)