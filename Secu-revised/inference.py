import argparse
import os
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
import shutil
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from munkres import Munkres
import secu
import secu.builder

import secu.folder
import secu.loader
import torchvision
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from nets.resnet_cifar import resnet18
import cv2
from PIL import Image
from scipy.optimize import linear_sum_assignment
from config import folder_path, clusters_amount, subfolder_path, result_Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 解析命令行參數
parser = argparse.ArgumentParser(description="SeCu Clustering Inference")
parser.add_argument("--batch-size", default=64, type=int, help="Mini-batch size (default: 512)")
parser.add_argument("--workers", default=4, type=int, help="Number of data loading workers (default: 8)")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument("--model-path", default=".\model\secu_0400.pth.tar",
                    type=str, help="Path to trained model checkpoint")
parser.add_argument('--data-name', default='cifar10', type=str,
                    help='name of data: cifar10, cifar100, stl10, custom')
parser.add_argument('--secu-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--secu-num-ins', default=50000, type=int,
                    help='number of instances (default: 50000)')
parser.add_argument('--secu-k', default=[8,9,10], type=int, nargs="+", help='multi-clustering head')
parser.add_argument('--secu-tx', default=0.05, type=float,
                    help='temperature for representation (default: 0.05)')
parser.add_argument('--secu-tw', default=0.05, type=float,
                    help='temperature for cluster center (default: 0.05)')
parser.add_argument('--secu-dual-lr', default=0.1, type=float,
                    help='dual learning rate for lower bound (default: 0.1)')
parser.add_argument('--secu-lratio', default=0.9, type=float,
                    help='lower-bound ratio (default: 0.4)')
parser.add_argument('--secu-alpha', default=6000, type=float,
                    help='entropy weight (default: 6000)')
parser.add_argument('--secu-cst', default='size', type=str,
                    help='constraint in secu: size or entropy')
parser.add_argument('--backbone', default='resnet18', type=str,
                    help='backbone architecture: resnet18 or vit')
parser.add_argument('--bands', default=3, type=int,
                    help='data type of tunnel number: 3 or 4 ')
parser.add_argument('--use-gradcam', default=False, type=bool,
                    help='use gradcam yes or not')
parser.add_argument('--data-path', 
                    default=r"E:\洋香瓜\adaptive_output", 
                    type=str, 
                    help='Path to dataset folder')

# 計算聚類績效
def cluster_metric(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    # pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = clustering_accuracy_score(label, pred)
    print("[Clustering Result]: ACC = {:.4f}, ARI = {:.4f}, NMI = {:.4f}".format(
        acc, ari, nmi
    ))
    return acc, ari, nmi



def clustering_accuracy_score(y_true, y_pred):
    """
    計算分群準確度 (ACC)。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    unique_true_labels = np.unique(y_true)  # 取得所有 Ground Truth 類別
    unique_pred_labels = np.unique(y_pred)  # 取得所有分群標籤

    cost_matrix = np.zeros((len(unique_true_labels), len(unique_pred_labels)))
    for i, true_label in enumerate(unique_true_labels):
        for j, pred_label in enumerate(unique_pred_labels):
            cost_matrix[i, j] = -np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    correct = 0
    for i in range(len(row_ind)):
        correct += np.sum((y_true == unique_true_labels[row_ind[i]]) & (y_pred == unique_pred_labels[col_ind[i]]))
    
    return correct / len(y_true)


def do_tsne(data, clusters, title):
    # 使用 t-SNE 進行降維
    tsne = TSNE(n_components=3, random_state=0)
    outputs_tsne = tsne.fit_transform(data)

    # 可視化原始 CIFAR-10 數據集的 t-SNE 結果
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 設置軸的限制範圍
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    ax.set_zlim((-20, 20))

    # 調整點的大小
    point_size = 10

    # 轉換類別字串為數字
    label_encoder = LabelEncoder()
    c_values = label_encoder.fit_transform(clusters)

    # 繪製 t-SNE 結果，使用不同顏色表示不同的簇
    scatter = ax.scatter(outputs_tsne[:, 0], outputs_tsne[:, 1], outputs_tsne[:, 2], 
                         c=c_values, cmap='viridis', s=point_size)

    # 添加圖例
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    plt.title(title)
    plt.show()


def calculate_cluster_label_distribution(predictions, ground_truth_labels, num_clusters):
    # 統計每個簇中類別的分佈
    cluster_label_counts = {}

    for cluster_idx in range(num_clusters):
        # 獲取該簇的資料索引
        cluster_indices = np.where(predictions == cluster_idx)[0]
        # 獲取該簇的標籤
        cluster_labels = ground_truth_labels[cluster_indices]
        # 統計該簇中各個類別的個數
        unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
        # 計算佔比
        label_percentages = label_counts / np.sum(label_counts)
        # 將統計結果保存到字典中
        cluster_label_counts[cluster_idx] = {
            'labels': unique_labels,
            'counts': label_counts,
            'percentages': label_percentages
        }
    return cluster_label_counts
'''
def create_distribution_dataframe(cluster_label_counts):
    # 創建一個空的列表來存儲類別分佈
    cluster_label_list = []

    # 循環每個簇中的類別分佈
    for cluster_idx, counts_info in cluster_label_counts.items():
        print(f"Cluster {cluster_idx} 的類別分佈：")
        # 按佔比大小排序
        sorted_indices = np.argsort(counts_info['percentages'])[::-1]
        sorted_labels = counts_info['labels'][sorted_indices]
        sorted_counts = counts_info['counts'][sorted_indices]
        sorted_percentages = counts_info['percentages'][sorted_indices]
        
        # 將數據存到列表中
        for label, count, percentage in zip(sorted_labels, sorted_counts, sorted_percentages):
            print(f"類別 {label}: 佔比 {percentage:.2%}")
            cluster_label_list.append({
                'Cluster': cluster_idx, 
                'Class': label, 
                'Count': count,
                'Percentage': f"{percentage:.0%}"  # 直接格式化為百分比字符串
            })

    # 用 pd.DataFrame() 直接轉換
    cluster_label_df = pd.DataFrame(cluster_label_list)

    return cluster_label_df'''
def create_distribution_dataframe(cluster_label_counts):
    # 創建一個空的 DataFrame 來存儲類別分佈
    cluster_label_list = []
    #計算每個類別的總數量
    count_class_number = {}
    for cluster_idx, counts_info in cluster_label_counts.items():
        # 按佔比大小排序
        sorted_indices = np.argsort(counts_info['percentages'])[::-1]
        sorted_labels = counts_info['labels'][sorted_indices]
        sorted_counts = counts_info['counts'][sorted_indices]
        sorted_percentages = counts_info['percentages'][sorted_indices]
        # 將數據添加到 DataFrame 中
        for label, count, percentage in zip(sorted_labels, sorted_counts, sorted_percentages):
            if label in count_class_number:
                count_class_number[label]+=count
            else:
                count_class_number[label]=count
    # 計算每個簇中的類別分佈
    for cluster_idx, counts_info in cluster_label_counts.items():
        # 按佔比大小排序
        sorted_indices = np.argsort(counts_info['percentages'])[::-1]
        sorted_labels = counts_info['labels'][sorted_indices]
        sorted_counts = counts_info['counts'][sorted_indices]
        sorted_percentages = counts_info['percentages'][sorted_indices]
        print(f"Cluster {cluster_idx} 的類別分佈：")
        print(f"Cluster {cluster_idx}'s total number：{sum(sorted_counts)}")
        # 將數據添加到 DataFrame 中
        for label, count, percentage in zip(sorted_labels, sorted_counts, sorted_percentages):
            print(f"類別 {label}: 佔比 {percentage:.2%},    數量:{count},    單一類別比例: {count/count_class_number[label]:.1%}")
            cluster_label_list.append({
                'Cluster': cluster_idx, 
                'Class': label, 
                'Count': count,
                'Percentage': f"{percentage:.0%}"  # 直接格式化為百分比字符串
            })

    cluster_label_df = pd.DataFrame(cluster_label_list)
    return cluster_label_df  

# 定義反向標準化函數
'''def unnormalize(image_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)  # 變成 (3, 1, 1) 用於廣播
    std = torch.tensor(std).view(3, 1, 1)    # 變成 (3, 1, 1) 用於廣播
    image_tensor = image_tensor * std + mean  # 反標準化
    image_tensor = torch.clamp(image_tensor, 0, 1)  # 限制範圍到 [0,1]
    return (image_tensor.numpy() * 255).astype(np.uint8)  # 轉為 NumPy 格式'''
def unnormalize(image_tensor, mean, std):
    image_tensor = image_tensor[:3]  # 只保留 RGB ✅
    mean = torch.tensor(mean[:3]).view(3, 1, 1)
    std = torch.tensor(std[:3]).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    return (image_tensor.numpy() * 255).astype(np.uint8)

def save_clustered_npy(combined_names, combined_list, predictions,all_file_path):
    os.makedirs('output',exist_ok=True)
    output_folder = os.path.join('output', "cluster "+str(clusters_amount))
    # 如果輸出資料夾已存在，刪除它
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # 創建輸出資料夾
    os.makedirs(output_folder)
    
    npy_folder = {}
    for cluster_id in set(predictions):
        npy_cluster_folder = os.path.join(output_folder,f'cluster_{cluster_id}')
        #創建放NPY 分群資料夾
        if not os.path.exists(npy_cluster_folder):
            os.makedirs(npy_cluster_folder)
        npy_folder[cluster_id] = npy_cluster_folder
    #print(all_file_path[0])
    all_file_path = sum(all_file_path, [])
    for (file_name, image, cluster_id) in zip(combined_names, all_file_path,predictions):
        # npy 找到對應的路徑
        #npy_cluster_folder = npy_folder[cluster_id]
        #matching_paths = [path for path in all_file_path if os.path.basename(path).split('.')[0] == file_name][0]
        npy_data = np.load(image)
        np.save(os.path.join(npy_folder[cluster_id],f'{file_name}.npy'),npy_data)


'''def save_clustered_images(file_names, image_list, predictions, output_folder):
    """
    將分群後的影像保存到不同的資料夾中。

    Args:
    - file_names (list): 包含影像檔案名稱的列表。
    - image_list (list): 包含影像數據的列表。
    - predictions (list): 包含每個影像所屬的 cluster 的列表。
    - output_folder (str): 輸出資料夾的路徑。

    Returns:
    - None
    """
    # 如果輸出資料夾已存在，刪除它
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # 創建輸出資料夾
    os.makedirs(output_folder)
    
    # 創建每個 cluster 的資料夾
    cluster_folders = {}
    for cluster_id in set(predictions):
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster_id}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        cluster_folders[cluster_id] = cluster_folder

    # 在保存影像之前，先 unnormalize
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    unnormalized_images = np.array([unnormalize(torch.tensor(img), mean, std) for img in image_list])

    # 將影像保存到對應的 cluster 資料夾中
    for (file_name, image, cluster_id) in zip(file_names, unnormalized_images, predictions):
        cluster_folder = cluster_folders[cluster_id]

        # 假設影像是 NumPy 陣列，且值範圍是 0-255
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # 確保是 (H, W, C) 形狀，如果是 (C, H, W)，需要轉置
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
            
        # 將 NumPy 陣列轉換為 PIL 圖像
        image_pil = Image.fromarray(image)
        image_path = os.path.join(cluster_folder, f'{file_name}.png')
        image_pil.save(image_path)'''


def save_clustered_images(file_names, file_paths, predictions, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    for f_name, src_path, cid in zip(file_names, file_paths, predictions):
        target_dir = os.path.join(output_folder, f'cluster_{cid}')
        os.makedirs(target_dir, exist_ok=True)
        ext = os.path.splitext(src_path)[1].lower()

        try:
            # --- 情況 A：如果是標準圖片格式，直接複製，不要用 np.load ---
            if ext in ['.jpg', '.jpeg', '.png']:
                dst_path = os.path.join(target_dir, f"{f_name}{ext}")
                shutil.copy(src_path, dst_path) 
                
            # --- 情況 B：如果是 npy 數據檔 ---
            elif ext == '.npy':
                dst_path = os.path.join(target_dir, f"{f_name}.png")
                # 關鍵修正：加入 allow_pickle=True
                data = np.load(src_path, allow_pickle=True)
                
                if data.ndim == 3:
                    if data.shape[0] in [3, 4]: # (C, H, W) -> (H, W, C)
                        data = data.transpose(1, 2, 0)
                    data = data[:, :, :3] # 只取 RGB
                
                # 數值範圍處理
                if data.dtype != np.uint8:
                    data = (data * 255).astype(np.uint8) if data.max() <= 1.5 else np.clip(data, 0, 255).astype(np.uint8)
                
                Image.fromarray(data).save(dst_path)
        except Exception as e:
            print(f"Error saving {f_name}: {e}")
    
'''def save_clustering_results_to_txt(names, predictions, output_dir):
    """
    Save clustering results grouped by frame into .txt files, sorted by the last number in the names.

    :param names: List of file names.
    :param predictions: List of clustering results corresponding to each file.
    :param output_dir: Directory to save the .txt files.
    """
    # 如果輸出資料夾已存在，刪除它
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # 創建輸出資料夾
    os.makedirs(output_dir)

    grouped_results = {}

    # 按圖框名稱分組
    for name, cluster in zip(names, predictions):
        frame_name = "_".join(name.split("_")[:-2])
        last_number = int(name.split("_")[-1])
        if frame_name not in grouped_results:
            grouped_results[frame_name] = []
        grouped_results[frame_name].append((last_number, cluster))

    # 保存結果到文件中
    for frame_name, clusters in grouped_results.items():
        clusters.sort(key=lambda x: x[0])

        sorted_clusters = [cluster for _, cluster in clusters]

        txt_file_path = os.path.join(output_dir, f"{frame_name}.txt")
        with open(txt_file_path, 'w') as f:
            f.write("\n".join(map(str, sorted_clusters)))'''
def save_clustering_results_to_txt(names, predictions, output_dir):
    """
    依 frame 分組，去掉最後兩段（時期 + 編號），
    依最後數字排序後，以「index,cluster」寫出（稀疏安全）。
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    grouped = defaultdict(list)

    for name, cluster in zip(names, predictions):
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"檔名格式太短：{name}")
        idx = int(re.search(r'\d+', parts[-1]).group())        # 檔名最後一段數字
        frame = "_".join(parts[:-2])  # 去掉「時期 + 數字」
        grouped[frame].append((idx, int(cluster)))

    for frame, items in grouped.items():
        items.sort(key=lambda x: x[0])  # 依 index 排序（可缺號）
        path = os.path.join(output_dir, f"{frame}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for idx, c in items:
                f.write(f"{idx},{c}\n")   # 關鍵：同時寫 index 與 cluster
class FourChannelTransform:
    def __init__(self, crop_size=224, mean=[0.4914, 0.4822, 0.4465, 0.5], std=[0.2023, 0.1994, 0.2010, 0.25]):
        self.crop_size = crop_size
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image):  # image is numpy array [H,W,C]
        if isinstance(image, Image.Image):
            image = np.array(image)  # PIL → NumPy

        # 然後轉為 PyTorch Tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()  # HWC → CHW
        image = torchvision.transforms.functional.center_crop(image, self.crop_size)
        image = image / 255.0
        return (image - self.mean) / self.std


'''class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        自定義 Dataset
        """
        self.transform = transform
        self.folder_path = folder_path
        
        # 掃描檔案並建立索引
        self.names, self.file_paths, self.labels, self.label2idx = self._scan_files(folder_path)

        print("-" * 30)
        print(f"Dataset 初始化完成: 共發現 {len(self.names)} 張影像")
        print(f"包含 {len(self.label2idx)} 個類別")
        print(f"類別映射表 (Label Map): {self.label2idx}")
        print("-" * 30)

    def _scan_files(self, folder_path):
        file_names = []
        file_paths = []
        ground_truth_list = [] 

        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            files.sort()
            
            # 1. 先把該資料夾下的所有 .npy 檔案找出來並排序
            npy_files = sorted([f for f in files if f.endswith('.jpg')])
            
            # 如果這個資料夾沒有 npy 檔，就跳過
            if not npy_files:
                continue

            # 2. 處理 GT_label.txt
            if 'GT_label.txt' in files:
                txt_path = os.path.join(root, 'GT_label.txt')
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        # --- 修正重點：讀取所有行，並去除空白行 ---
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        
                    # --- 檢查數量是否匹配 ---
                    if len(lines) != len(npy_files):
                        print(f"[Warning] 數量不符！資料夾 {os.path.basename(root)} 中：")
                        print(f"   .npy 檔案數: {len(npy_files)}")
                        print(f"   Label 行數 : {len(lines)}")
                        print("   -> 將嘗試以最小長度進行配對，可能會遺失資料。")
                        min_len = min(len(lines), len(npy_files))
                        lines = lines[:min_len]
                        npy_files = npy_files[:min_len]

                    # --- 一對一配對 ---
                    for i, file in enumerate(npy_files):
                        full_path = os.path.join(root, file)
                        f_name = os.path.splitext(file)[0]
                        label_name = lines[i]  # 取得對應的那一行標籤

                        file_names.append(f_name)
                        file_paths.append(full_path)
                        ground_truth_list.append(label_name)

                except Exception as e:
                    print(f"[Error] 讀取 {txt_path} 失敗: {e}")
                    continue
            else:
                # 如果沒有 GT_label.txt，視需求決定是否跳過
                # 這裡假設沒標籤就不讀圖
                continue

        if not file_names:
            print("[Error] 未找到任何有效的資料！")
            return [], [], [], {}

        # 整理與建立索引
        zipped = sorted(zip(file_names, file_paths, ground_truth_list), key=lambda x: x[0])
        file_names, file_paths, ground_truth_list = zip(*zipped)
        
        file_names = list(file_names)
        file_paths = list(file_paths)
        ground_truth_list = list(ground_truth_list)

        unique_labels = sorted(list(set(ground_truth_list)))
        label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label2idx[label] for label in ground_truth_list]

        return file_names, file_paths, labels, label2idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        修正：
        1. 確保回傳 4 個值 (name, tensor, label, path)
        2. 強制處理 4 通道 (RGBA) 轉為 3 通道 (RGB)
        """
        npy_path = self.file_paths[idx]
        label_id = self.labels[idx]
        # 提取檔名 (不含路徑)，用於回傳第一個參數 names
        file_name = os.path.basename(npy_path)

        try:
            # 1. 讀取 npy (ndarray)
            img_data = np.load(npy_path)
            
            # --- 通道數修正：解決 Tensor a(4) vs b(3) 的問題 ---
            # 如果 numpy array 是 (H, W, 4)，強制取前 3 個通道 (RGB)
            if isinstance(img_data, np.ndarray) and len(img_data.shape) == 3:
                if img_data.shape[2] == 4:
                    img_data = img_data[:, :, :3]
                elif img_data.shape[0] == 4: # 有些人習慣放 (C, H, W)
                    img_data = img_data[:3, :, :].transpose(1, 2, 0) # 轉回 PIL 吃的 (H, W, C)

            # --- 影像轉換邏輯 ---
            if self.transform:
                if isinstance(img_data, np.ndarray):
                    # 1. 確保數據類型是 uint8
                    if img_data.dtype != np.uint8:
                        if img_data.max() <= 1.5: 
                            img_data = (img_data * 255).astype(np.uint8)
                        else:
                            img_data = np.clip(img_data, 0, 255).astype(np.uint8)
                    
                    # 2. 轉成 PIL Image
                    try:
                        # 再次確保轉成 RGB 模式 (排除 RGBA 殘留)
                        img_data = Image.fromarray(img_data).convert('RGB')
                    except Exception as e:
                        print(f"Warning: Failed to convert {npy_path} to PIL: {e}")

                # 3. 執行 transform
                img_tensor = self.transform(img_data)
            else:
                # 如果沒有 transform，手動轉為 Tensor 並調整維度為 (C, H, W)
                img_tensor = torch.from_numpy(img_data).float()
                if img_tensor.ndimension() == 3 and img_tensor.shape[2] == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)

            # --- 關鍵修正：回傳 4 個值，對應 (names, images, targets, file_paths) ---
            return file_name, img_tensor, label_id, npy_path

        except Exception as e:
            print(f"[Error] loading file {npy_path}: {e}")
            # 發生錯誤時也必須回傳 4 個值，避免 DataLoader 崩潰
            dummy_tensor = torch.zeros(3, 224, 224)
            return file_name, dummy_tensor, label_id, npy_path'''

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.folder_path = folder_path
        
        # 建立類別映射表：資料夾名稱 -> 數字索引
        self.names, self.file_paths, self.ground_truth_names, self.label2idx = self._scan_files(folder_path)
        
        # 根據映射表將字串類別轉成數字 ID
        self.labels = [self.label2idx[name] for name in self.ground_truth_names]

        print("-" * 30)
        print(f"Dataset 初始化完成: 共發現 {len(self.names)} 張影像")
        print(f"包含 {len(self.label2idx)} 個類別")
        print(f"類別映射表 (Label Map): {self.label2idx}")
        print("-" * 30)

    def _scan_files(self, folder_path):
        file_names = []
        file_paths = []
        ground_truth_list = [] 
        
        # 支援的格式
        valid_extensions = ('.npy', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

        # 遍歷所有子資料夾，資料夾名稱就是 Label
        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            
            # 找出目前資料夾中所有符合格式的檔案
            current_files = sorted([f for f in files if f.lower().endswith(valid_extensions)])
            
            if not current_files:
                continue

            # 取得目前的資料夾名稱作為類別 (Label)
            label_name = os.path.basename(root)

            for file in current_files:
                full_path = os.path.join(root, file)
                f_name = os.path.splitext(file)[0]

                file_names.append(f_name)
                file_paths.append(full_path)
                ground_truth_list.append(label_name)

        if not file_names:
            print("[Error] 未找到任何有效的資料！")
            return [], [], [], {}

        # 建立類別映射 (例如: {'cat': 0, 'dog': 1})
        unique_labels = sorted(list(set(ground_truth_list)))
        label2idx = {label: idx for idx, label in enumerate(unique_labels)}

        return file_names, file_paths, ground_truth_list, label2idx

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label_id = self.labels[idx]
        file_name = os.path.basename(file_path)
        
        # 取得副檔名並轉小寫
        ext = os.path.splitext(file_path)[1].lower()

        try:
            # --- 分流處理：根據副檔名選擇讀取工具 ---
            if ext == '.npy':
                # 處理 numpy 檔案
                img_data = np.load(file_path, allow_pickle=True)
                
                # 確保轉成 (H, W, C) 格式給 PIL
                if img_data.ndim == 3:
                    if img_data.shape[0] == 3 or img_data.shape[0] == 4:
                        img_data = img_data.transpose(1, 2, 0)
                
                # 處理 4 通道轉 3 通道
                if img_data.shape[-1] == 4:
                    img_data = img_data[:, :, :3]
                
                # 確保數值範圍與型別
                if img_data.dtype != np.uint8:
                    img_data = (img_data * 255).astype(np.uint8) if img_data.max() <= 1.5 else np.clip(img_data, 0, 255).astype(np.uint8)
                
                img_pil = Image.fromarray(img_data).convert('RGB')

            elif ext in ['.jpg', '.jpeg', '.png']:
                # --- 這是解決你報錯的關鍵：直接用 PIL 讀取圖片 ---
                img_pil = Image.open(file_path).convert('RGB')
            
            else:
                raise ValueError(f"不支援的檔案格式: {ext}")

            # --- 統一進入 Transform ---
            if self.transform:
                img_tensor = self.transform(img_pil)
            else:
                # 沒 transform 就手動轉 tensor
                img_array = np.array(img_pil)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

            return file_name, img_tensor, label_id, file_path

        except Exception as e:
            print(f"[Error] 讀取檔案失敗 {file_path}: {e}")
            # 回傳空 Tensor 防止 DataLoader 斷掉
            dummy_tensor = torch.zeros(3, 224, 224)
            return file_name, dummy_tensor, label_id, file_path

    def __len__(self):
        return len(self.file_paths)

class FeatureHook:
    def __init__(self, layer):
        self.feature = None
        self.grad = None
        layer.register_forward_hook(self.forward_hook)
        layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feature = output

    def backward_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0]


def compute_gradcam(model, img_tensor):
    model.zero_grad()

    conv_out = None
    conv_grad = None

    # ResNet 最後一層 conv
    last_conv = model.encoder.layer4[-1].conv2

    def forward_hook(module, inp, out):
        nonlocal conv_out
        conv_out = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal conv_grad
        conv_grad = grad_out[0]

    last_conv.register_forward_hook(forward_hook)
    last_conv.register_full_backward_hook(backward_hook)

    img_tensor.requires_grad_(True)

    # ------------------------------------------------
    # 手動 forward → 取得 CNN feature
    # ------------------------------------------------
    x = model.encoder.conv1(img_tensor)
    x = model.encoder.bn1(x)
    x = F.relu(x)
    x = model.encoder.layer1(x)
    x = model.encoder.layer2(x)
    x = model.encoder.layer3(x)
    x = model.encoder.layer4(x)   # ⭐ conv_out 在這裡被 hook 到

    feat_map = x                      # shape = (1, 512, 7, 7)
    gap = F.adaptive_avg_pool2d(x, 1) # shape = (1, 512, 1, 1)
    gap = torch.flatten(gap, 1)       # shape = (1, 512)

    # ------------------------------------------------
    # projector = encoder.fc
    # ------------------------------------------------
    proj = model.encoder.fc(gap)      # shape (1, 128)
    loss = proj.norm()
    loss.backward()

    # ------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------
    weights = conv_grad.mean(dim=(2,3), keepdim=True)
    cam = (weights * conv_out).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam /= (cam.max() + 1e-7)

    return cam.detach().cpu().numpy()



def overlay_cam_on_image(image_np, cam, alpha=0.5):
    H, W = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (image_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay

def save_gradcam(raw_np, cam, cluster_id, img_name):
    save_dir = f"data/test/grad_cam/cluster_{cluster_id}"
    os.makedirs(save_dir, exist_ok=True)

    cam_resized = cv2.resize(cam, (raw_np.shape[1], raw_np.shape[0]))
    cam_norm = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.4 * raw_np + 0.6 * heatmap).astype(np.uint8)

    out_path = os.path.join(save_dir, f"{img_name}_cam.png")
    Image.fromarray(overlay).save(out_path)

def main():
    args = parser.parse_args()

    # 設定 GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # **載入模型**
    print("=> Creating model and loading checkpoint...")
    if args.backbone=='resnet18':
        if args.data_name == 'cifar10':
            from nets.resnet_cifar import resnet18
            base_encoder = resnet18
        elif args.data_name == 'stl10':
            from nets.resnet_stl import resnet18
            base_encoder = resnet18
        elif args.data_name=='cifar10' or args.data_name=='cifar100':
            from nets.resnet_cifar import resnet18
            base_encoder = resnet18
        elif args.data_name == 'custom':
            from nets.resnet_custom import resnet18
            base_encoder = resnet18

    elif args.backbone == 'vit':
        from nets.vit import ViT
        base_encoder = ViT
    elif args.backbone == 'dinov2':
        from nets.vit import DINOv2
        base_encoder = DINOv2
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    model = secu.builder.SeCu(
        base_encoder=base_encoder,
        K=args.secu_k,
        tx=args.secu_tx,
        tw=args.secu_tw,
        dim=args.secu_dim,
        num_ins=args.secu_num_ins,
        alpha=args.secu_alpha,
        dual_lr=args.secu_dual_lr,
        lratio=args.secu_lratio,
        constraint=args.secu_cst
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model.eval()

    # **載入預訓練權重**
    checkpoint = torch.load(args.model_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[Debug] 缺少的層數: {len(missing)}")
    print(f"[Debug] 多餘的層數: {len(unexpected)}")
    print(f"[Debug] 缺少的層 (前10個): {missing[:10]}")
    model.load_state_dict(new_state_dict, strict=False)
    model.load_param()

    print("Model loaded successfully!")

    # **設定影像轉換**
    if args.bands == 4:
        
        nir_mean,nir_std = secu.loader.compute_mean_std_npy(r'data/train')
        rgb_mean=[0.4914, 0.4822, 0.4465]
        rgb_std=[0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean=rgb_mean+[nir_mean], std=rgb_std+[nir_std])
        transform = secu.loader.RGBNIRTransform(train=False,nir_mean=nir_mean, nir_std=nir_std)
    elif args.bands == 3:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
        transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),normalize])
        #transform = transforms.Compose([transforms.Resize((224, 224)),normalize])
    
    # 讀取 cifar_test.txt，並用 DataLoader 批次載入
    testdir = args.data_path

    # 為了確保萬無一失，在下面多加這一行偵錯，執行時我們就能看到路徑對不對：
    print(f"\n[偵錯] 正在掃描路徑：{os.path.abspath(testdir)}")

    # 加上這幾行來診斷：
    print(f"\n[偵錯] 目前指定的路徑是: {testdir}")
    if os.path.exists(testdir):
        print(f"[偵錯] 路徑確認存在！")
        subfolders = [f for f in os.listdir(testdir) if os.path.isdir(os.path.join(testdir, f))]
        print(f"[偵錯] 該路徑下發現的子資料夾數量: {len(subfolders)}")
        if len(subfolders) > 0:
            print(f"[偵錯] 第一個子資料夾名稱: {subfolders[0]}")
    else:
        print(f"[偵錯] 警告！路徑不存在，請檢查拼字。")
    # ------------------
    print(f"\n[Debug] 程式目前正在掃描這個路徑: {os.path.abspath(testdir)}")
    if args.data_name == 'custom':
        test_dataset = ImageDataset(testdir, transform=transform)
    elif args.data_name == 'cifar10':
        from torchvision.datasets import ImageFolder
        test_dataset = ImageFolder(testdir, transform=transform)
        #test_dataset = CIFAR10(root=testdir,train=False,transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


    # **進行推論**
    print("Running inference...")
    predictions = []
    labels_test = []
    combined_names = []
    all_files_path = []     # 新增：用來存路徑
    combined_outputs = []   # 用來存 Feature (for t-SNE)
    with torch.no_grad():
        if args.data_name == 'custom':
            for names, images, targets, file_paths in test_loader:
                images = images.to(device, non_blocking=True)
                
                # 取得預測
                output = model.get_pred(images)
                feature = model.get_feature(images)
                
                # 存結果
                predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                labels_test.append(targets.cpu().numpy())
                combined_outputs.append(feature.cpu().numpy())
                
                # 存檔名與路徑
                combined_names.extend(names)
                all_files_path.extend(file_paths) # 存路徑字串，記憶體消耗極小
                
                
        elif args.data_name == 'cifar10':
            # 針對 CIFAR10 的處理 (通常用於測試)
            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(device)
                output = model.get_pred(images)
                feature = model.get_feature(images)

                predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                labels_test.append(targets.cpu().numpy())
                combined_outputs.append(feature.cpu().numpy())

                # CIFAR 沒有路徑，生成假路徑 (注意：這樣無法使用 save_clustered_images)
                batch_size = images.size(0)
                fake_names = [f"cifar10_{batch_idx*args.batch_size+i}" for i in range(batch_size)]
                combined_names.extend(fake_names)
                all_files_path.extend(["none" for _ in range(batch_size)])
    
    
    
    predictions = np.concatenate(predictions)
    labels_test = np.concatenate(labels_test)
    combined_outputs = np.concatenate(combined_outputs) # (N, Dim)

    if args.use_gradcam:
        print("Generating Grad-CAM images...")
        gradcam_base = "data/test/grad_cam"
        os.makedirs(gradcam_base, exist_ok=True)

        for i in range(len(combined_names)):
            img_path = all_files_path[i]
            img_name = combined_names[i]
            cluster_id = predictions[i]
            ext = os.path.splitext(img_path)[1].lower()

            try:
                # --- ★ 核心修正：Grad-CAM 影像讀取分流 ---
                if ext == '.npy':
                    # NPY 才用 np.load 並開啟 allow_pickle
                    raw_np = np.load(img_path, allow_pickle=True)
                    if raw_np.ndim == 3 and raw_np.shape[0] in [3, 4]:
                        raw_np = raw_np.transpose(1, 2, 0)
                    raw_np = raw_np[:, :, :3]
                else:
                    # JPG/PNG 使用 PIL 讀取
                    raw_np = np.array(Image.open(img_path).convert('RGB'))

                # 統一處理數據格式
                raw_np = np.clip(raw_np, 0, 255).astype(np.uint8)

                # 生成模型所需的 tensor
                pil_img = Image.fromarray(raw_np)
                # 使用 transform 以確保與訓練時的標準化一致 (Resize, ToTensor, Normalize)
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                img_tensor.requires_grad_(True)

                cam = compute_gradcam(model, img_tensor)
                save_gradcam(raw_np, cam, cluster_id, img_name)
            except Exception as e:
                print(f"Error generating Grad-CAM for {img_name}: {e}")

        print(f"Total samples processed for Grad-CAM: {len(predictions)}")
    if args.data_name == 'custom':
        save_clustered_images(combined_names, all_files_path, predictions, subfolder_path)
    
    # **計算聚類績效**
    acc_score, ari_score, nmi_score = cluster_metric(labels_test, predictions)

    # t-SNE 可視化
    do_tsne(combined_outputs, labels_test, "t-SNE Visualization of Ground Truth")
    do_tsne(combined_outputs, predictions, "t-SNE Visualization of cluster result")

    # 統計每個簇中的類別分佈
    cluster_label_counts = calculate_cluster_label_distribution(predictions, labels_test, num_clusters=clusters_amount)
    cluster_label_df = create_distribution_dataframe(cluster_label_counts)
    pivot_table = cluster_label_df.pivot_table(index='Class', columns='Cluster', values='Percentage', aggfunc='first')

    # 補上指標分數
    metrics_df = pd.DataFrame({
        'ACC': [f"{acc_score:.4f}"],
        'ARI': [f"{ari_score:.4f}"],
        'NMI': [f"{nmi_score:.4f}"]
    }, index=['Score']).T
    
    # 這裡簡單處理，將分數 print 出來或手動加到 CSV 底部 (視你的 pandas 版本處理方式而定)
    # 為了簡單起見，直接存主要的 pivot table
    csv_path = os.path.join(subfolder_path, 'cluster_label_distribution.csv')
    pivot_table.to_csv(csv_path, encoding="utf-8-sig")
    
    # 另外存一個分數檔，或者 append 寫入
    with open(csv_path, 'a') as f:
        f.write(f"\nACC,{acc_score:.4f}\nARI,{ari_score:.4f}\nNMI,{nmi_score:.4f}\n")

    print(f"CSV file created at: {csv_path}")
    
    # 存 txt 結果
    save_clustering_results_to_txt(combined_names, predictions, result_Path)
    print(f"Clustering results .txt written to: {result_Path}")

    '''from sklearn.cluster import KMeans
    real_k = 12
    print(f"Running K-Means with K={real_k} ...")

    # 2. 執行 K-Means
    # n_init=20 代表隨機跑 20 次取最好的結果，避免掉入局部最佳解
    kmeans = KMeans(n_clusters=real_k, random_state=0, n_init=200)
    kmeans_pred = kmeans.fit_predict(combined_outputs)
    save_clustered_images(combined_names, all_files_path, kmeans_pred, subfolder_path+'_kmeans')
    # 3. 計算 K-Means 的績效 (ACC, ARI, NMI)
    print(f"--- K-Means Result (K={real_k}) ---")
    acc_km, ari_km, nmi_km = cluster_metric(labels_test, kmeans_pred)

    # 4. K-Means 的 t-SNE 可視化
    # 這樣你就可以比較 SeCu 預測的圖 vs K-Means 預測的圖
    do_tsne(combined_outputs, kmeans_pred, f"t-SNE Visualization (K-Means Result)")
    print("Generating K-Means distribution report...")
    km_cluster_counts = calculate_cluster_label_distribution(kmeans_pred, labels_test, num_clusters=real_k)
    km_df = create_distribution_dataframe(km_cluster_counts)
    
    # 6. 儲存 K-Means 的 CSV 結果
    km_csv_path = os.path.join(subfolder_path, 'kmeans_distribution.csv')
    km_pivot = km_df.pivot_table(index='Class', columns='Cluster', values='Percentage', aggfunc='first')
    km_pivot.to_csv(km_csv_path, encoding="utf-8-sig")'''
if __name__ == '__main__':
    main()
