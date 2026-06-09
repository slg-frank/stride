
import argparse
import json
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import numpy as np

from imad_v2_model import IMAD_V2



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

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    node_num = metadata['node_num']
    edges = tuple(map(tuple, metadata['edges']))
    event_num = metadata['event_num']
    metric_num_config = metadata.get('metric_num', 7)


    first_chunk_id = list(data.keys())[0]
    first_chunk = data[first_chunk_id]

    actual_metric_num = first_chunk['metrics'].shape[-1]
    actual_event_num = first_chunk['logs'].shape[-1]


    if metric_num_config != actual_metric_num:
        metric_num = actual_metric_num
    else:
        metric_num = metric_num_config

    if event_num != actual_event_num:
        event_num = actual_event_num

    return data, node_num, edges, event_num, metric_num


def load_teacher_embeddings(embedding_path):
    """加载LLM teacher embeddings"""
    if not os.path.exists(embedding_path):
        return None

    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    return torch.FloatTensor(embeddings)


def evaluate(model, dataloader, device, phase="Val"):
    """评估模型"""
    model.eval()

    all_detect_preds = []
    all_detect_labels = []
    all_locate_logits = []
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
    random_seed = get_config('random_seed', 46)
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # 设置结果保存目录
    result_dir = get_config('result_dir', './results/imad_v2')
    os.makedirs(result_dir, exist_ok=True)

    # 设置设备
    use_gpu = get_config('use_gpu', True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_data, node_num, edges, event_num, metric_num = load_data(
        get_config('train_data_path'),
        get_config('metadata_path')
    )
    test_data, _, _, _, _ = load_data(
        get_config('test_data_path'),
        get_config('metadata_path')
    )


    # 加载teacher embeddings
    teacher_embeddings = None
    if get_config('use_imad_v2', True):
        embedding_path = get_config('teacher_embedding_path')
        if embedding_path:
            teacher_embeddings = load_teacher_embeddings(embedding_path)
            if teacher_embeddings is not None:
                teacher_embeddings = teacher_embeddings.to(device)

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


    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=get_config('lr', 0.001))

    # 训练
    best_f1 = 0
    best_hr1 = 0
    patience_counter = 0
    patience = get_config('patience', 15)
    epochs = get_config('epochs', 125)
    eval_every = get_config('eval_every', 5)



    # 保存结果
    with open(os.path.join(result_dir, 'final_results.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IMAD_V2 model')
    parser.add_argument('--config', type=str, default='config_imad_v2.json',
                        help='Path to config file')
    args = parser.parse_args()

    main(args.config)
