import argparse
import os
import torch
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
import secu.builder_imagenet
import secu.folder
import torchvision
from torch.utils.data import Dataset, DataLoader
from nets.resnet_cifar import resnet18
from PIL import Image
from scipy.optimize import linear_sum_assignment
from config import folder_path, clusters_amount, subfolder_path, result_Path

# 解析命令行參數
parser = argparse.ArgumentParser(description="SeCu Clustering Inference")
parser.add_argument("--batch-size", default=32, type=int, help="Mini-batch size (default: 512)")
parser.add_argument("--workers", default=4, type=int, help="Number of data loading workers (default: 8)")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument("--model-path", default=".\model\secu_0400.pth.tar",
                    type=str, help="Path to trained model checkpoint")

parser.add_argument('--secu-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--secu-num-ins', default=50000, type=int,
                    help='number of instances (default: 50000)')
parser.add_argument('--secu-k', default=[3,4,5], type=int, nargs="+", help='multi-clustering head')
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
def unnormalize(image_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)  # 變成 (3, 1, 1) 用於廣播
    std = torch.tensor(std).view(3, 1, 1)    # 變成 (3, 1, 1) 用於廣播
    image_tensor = image_tensor * std + mean  # 反標準化
    image_tensor = torch.clamp(image_tensor, 0, 1)  # 限制範圍到 [0,1]
    return (image_tensor.numpy() * 255).astype(np.uint8)  # 轉為 NumPy 格式

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


    
    
def save_clustered_images(file_names, image_list, predictions, output_folder):
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
        image_pil.save(image_path)


def save_clustering_results_to_txt(names, predictions, output_dir):
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
        frame_name = "_".join(name.split("_")[:-1])
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
            f.write("\n".join(map(str, sorted_clusters)))


class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        自定義 Dataset 來讀取 .npy 影像數據及其對應的 Ground Truth 標籤。

        Args:
        - folder_path (str): 包含 .npy 檔案與 ground truth 檔案的資料夾。
        - transform (callable, optional): 影像變換函數。
        """
        self.transform = transform
        self.names = []
        self.data = []
        self.labels = []
        self.file_path = []

        # 讀取 .npy 影像數據
        file_names, arrays_list,ground_truth_list,file_path = self.read_npy_files(folder_path)
        #ground_truth_list = self.read_ground_truth(folder_path)

        # 檢查檔案數量是否一致
        assert len(file_names) == len(ground_truth_list), "影像數據與 Ground Truth 數量不匹配！"

        # 存入 data 與 labels
        self.names = file_names
        self.data = arrays_list
        self.file_path =file_path

        label2idx = {label: idx for idx, label in enumerate(sorted(set(ground_truth_list)))}
        self.labels = [label2idx[label] for label in ground_truth_list]
        #self.labels = [int(label) for label in ground_truth_list]  # 轉為整數類別

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name = self.names[index]
        image = self.data[index]
        label = self.labels[index]
        file_path =self.file_path[index]

        # 轉換 NumPy 陣列為 PIL Image 或 Tensor
        image = Image.fromarray(image.astype('uint8'))

        if self.transform:
            image = self.transform(image)  # 進行影像變換
        
        return name, image, label,file_path  # 返回影像與標籤

    def read_npy_files(self, folder_path):
        """ 讀取資料夾中的 .npy 影像數據 """
        file_names = []
        arrays_list = []
        ground_truth_list = []
        all_file_path = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    all_file_path.append(file_path)
                    array = np.load(file_path)[:, :, :3]  # 只保留前三個通道
                    arrays_list.append(array)
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    file_names.append(file_name)
                    gt_name = file_name.split('_')[7]
                    ground_truth_list.append(gt_name)

        return file_names, arrays_list,ground_truth_list,all_file_path

    def read_ground_truth(self, folder_path):
        """ 讀取 Ground Truth 標籤 """
        ground_truth_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('GT_label.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding="utf-8") as f:
                            lines = f.readlines()
                        ground_truth_list.extend([line.strip() for line in lines])
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='big5') as f:
                            lines = f.readlines()
                        ground_truth_list.extend([line.strip() for line in lines])
        return ground_truth_list
    
def main():
    args = parser.parse_args()

    # 設定 GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # **載入模型**
    print("=> Creating model and loading checkpoint...")
    model = secu.builder.SeCu(
        base_encoder=resnet18,
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
    model.load_state_dict(new_state_dict, strict=False)
    model.load_param()

    print("Model loaded successfully!")

    # **設定影像轉換**
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 讀取 cifar_test.txt，並用 DataLoader 批次載入
    testdir = folder_path
    test_dataset = ImageDataset(testdir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


    # **進行推論**
    print("Running inference...")
    predictions = []
    probs = []
    labels_test = []
    combined_names, combined_list, combined_outputs = [], [], []
    all_files_path = []
    with torch.no_grad():
        for names, images, targets ,file_path in test_loader:
            images = images.to(device)
            output = model.get_pred(images)  # 取得預測標籤
            predictions.append(torch.argmax(output, dim=1).cpu().numpy())
            probs.append(F.softmax(output, dim=1).cpu().numpy())
            labels_test.append(targets.cpu().numpy())
            combined_names.extend(names)
            combined_list.append(images.cpu())
            feature = model.get_feature(images)
            combined_outputs.append(feature.cpu())
            all_files_path.append(file_path)

    combined_list = torch.cat(combined_list, dim=0).cpu().numpy()
    combined_outputs = torch.cat(combined_outputs, dim=0).cpu().numpy()

    # **整理結果**
    predictions = np.concatenate(predictions)
    probs = np.concatenate(probs)
    labels_test = np.concatenate(labels_test)
    
    # 保存分群後的影像
    save_clustered_images(combined_names, combined_list, predictions, subfolder_path)

    # 保存分群後的 .npy (便於後續分析)
    save_clustered_npy(combined_names, combined_list, predictions,all_files_path)
    # **計算聚類績效**
    acc_score, ari_score, nmi_score = cluster_metric(labels_test, predictions)

    # t-SNE 可視化
    do_tsne(combined_outputs, labels_test, "t-SNE Visualization of Ground Truth")
    do_tsne(combined_outputs, predictions, "t-SNE Visualization of cluster result")

    # 統計每個簇中的類別分佈
    cluster_label_counts = calculate_cluster_label_distribution(predictions, np.array(labels_test), num_clusters=clusters_amount)

    # 創建包含類別分佈的 DataFrame
    cluster_label_df = create_distribution_dataframe(cluster_label_counts)

    # 使用 pivot_table 函數將 DataFrame 重塑為矩陣形式
    pivot_table = cluster_label_df.pivot_table(index='Class', columns='Cluster', values='Percentage', aggfunc='first')

    # 新增一行 ACC，填入最後一欄
    acc_row = {col: '' for col in pivot_table.columns}  # 其他欄位設為空
    acc_row[pivot_table.columns[-1]] = f"{acc_score:.4f}"  # 在最後一欄填入 ACC 值
    acc_row_df = pd.DataFrame([acc_row], index=['ACC'])

    # 新增一行 ARI，填入最後一欄
    ari_row = {col: '' for col in pivot_table.columns}  # 其他欄位設為空
    ari_row[pivot_table.columns[-1]] = f"{ari_score:.4f}"  # 在最後一欄填入 ARI 值
    ari_row_df = pd.DataFrame([ari_row], index=['ARI'])

    # 新增一行 NMI，填入最後一欄
    nmi_row = {col: '' for col in pivot_table.columns}  # 其他欄位設為空
    nmi_row[pivot_table.columns[-1]] = f"{nmi_score:.4f}"  # 在最後一欄填入 ARI 值
    nmi_row_df = pd.DataFrame([nmi_row], index=['NMI'])

    # 將 ACC 行加入到 DataFrame
    pivot_table = pd.concat([pivot_table, acc_row_df])

    # 將 ARI 行加入到 DataFrame
    pivot_table = pd.concat([pivot_table, ari_row_df])

    # 將 NMI 行加入到 DataFrame
    pivot_table = pd.concat([pivot_table, nmi_row_df])

    # 將 DataFrame 保存為 CSV 文件
    pivot_table.to_csv(os.path.join(subfolder_path, 'cluster_label_distribution.csv'), encoding="utf-8-sig")

    print("CSV file 'cluster_label_distribution.csv' has been created.")

    # 將分群結果寫進 txt 中
    #save_clustering_results_to_txt(combined_names, predictions, result_Path)

    print(f"Clustering results .txt have been successfully written to: {result_Path}")


if __name__ == '__main__':
    main()
