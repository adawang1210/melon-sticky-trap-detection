"""
subcluster.py — 對單一 cluster 進行子聚類（Hierarchical Clustering）

使用 frozen DINOv2 patch-level 特徵 + HSV 顏色直方圖 + 大小/面積特徵，
支援 HDBSCAN（自動決定 K）和 K-Means 兩種聚類方法。

用法：
  # HDBSCAN（自動決定子群數，推薦）
  python subcluster.py -i <cluster資料夾> --method hdbscan --preview
  python subcluster.py -i <父資料夾> --all --method hdbscan --preview

  # K-Means（手動指定 K）
  python subcluster.py -i <cluster資料夾> --method kmeans -k 2 3 4 5 --preview

  # 調整特徵權重
  python subcluster.py -i <cluster資料夾> --method hdbscan --color-weight 2.0 --size-weight 1.5

輸出：
  <input>_subcluster/
  ├── hdbscan/          或 k3/
  │   ├── sub_0/
  │   ├── sub_1/
  │   └── noise/        ← HDBSCAN 的噪點（K-Means 沒有）
  └── hdbscan_preview/  或 k3_preview/
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


# ============================================================
# 特徵提取
# ============================================================

def collect_images(input_dir: Path):
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )
    if not images:
        raise FileNotFoundError(f"在 '{input_dir}' 找不到任何圖片")
    log.info("找到 %d 張圖片", len(images))
    return images


def extract_patch_features(image_paths, device="cuda", batch_size=32):
    """用 frozen DINOv2 提取 patch-level 特徵（mean + std = 768×2 = 1536 維）。

    相比 CLS token（768 維），patch-level 特徵保留了更多空間細節，
    對區分形態相似但細節不同的昆蟲更有效。
    """
    import timm

    from torchvision import transforms

    log.info("載入 DINOv2 模型（patch-level 特徵）...")
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=True,
        num_classes=0,
        img_size=224,
    )
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    log.info("提取 patch-level 特徵中（batch_size=%d）...", batch_size)

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))

            batch = torch.stack(batch_tensors).to(device)

            # 取得所有 patch tokens（不只 CLS token）
            # forward_features 回傳 (B, num_patches+1, 768)，第 0 個是 CLS
            x = model.forward_features(batch)
            patch_tokens = x[:, 1:, :]  # 去掉 CLS，只留 patch tokens

            # 計算 mean 和 std 作為特徵
            patch_mean = patch_tokens.mean(dim=1)  # (B, 768)
            patch_std = patch_tokens.std(dim=1)    # (B, 768)
            feat = torch.cat([patch_mean, patch_std], dim=1)  # (B, 1536)

            all_features.append(feat.cpu().numpy())

            if (i // batch_size) % 5 == 0:
                log.info("  進度：%d / %d", min(i + batch_size, len(image_paths)), len(image_paths))

    features = np.concatenate(all_features, axis=0)
    log.info("Patch-level 特徵 shape: %s", features.shape)
    return features


def extract_color_features(image_paths, bins=64):
    """提取前景區域的 HSV 顏色特徵（排除黃色背景）。

    特徵包含：
    - 前景 HSV 直方圖（64 bins × 3 通道 = 192 維）
    - 前景顏色統計量（mean H/S/V + std H/S/V + peak H = 7 維）
    總共 199 維
    """
    log.info("提取前景顏色特徵（排除黃色背景，bins=%d）...", bins)
    all_feats = []
    for p in image_paths:
        img = Image.open(p).convert("HSV")
        arr = np.array(img, dtype=np.uint8)
        h, s, v = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # 前景 mask：排除黃色背景和黑色區域
        is_yellow = (h >= 20) & (h <= 55) & (s >= 40) & (v >= 80)
        is_black = v <= 30
        foreground = ~is_yellow & ~is_black

        if foreground.sum() < 10:
            # 前景太少，用整張圖（避免空特徵）
            fg_h, fg_s, fg_v = h.ravel(), s.ravel(), v.ravel()
        else:
            fg_h = h[foreground]
            fg_s = s[foreground]
            fg_v = v[foreground]

        # 前景 HSV 直方圖
        h_hist, _ = np.histogram(fg_h, bins=bins, range=(0, 255))
        s_hist, _ = np.histogram(fg_s, bins=bins, range=(0, 255))
        v_hist, _ = np.histogram(fg_v, bins=bins, range=(0, 255))
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)

        # 前景顏色統計量
        mean_h = fg_h.mean() / 255.0
        mean_s = fg_s.mean() / 255.0
        mean_v = fg_v.mean() / 255.0
        std_h = fg_h.std() / 255.0
        std_s = fg_s.std() / 255.0
        std_v = fg_v.std() / 255.0
        # 主色調：H 直方圖的 peak 位置
        peak_h = h_hist.argmax() / bins

        stats = np.array([mean_h, mean_s, mean_v, std_h, std_s, std_v, peak_h],
                         dtype=np.float32)

        feat = np.concatenate([hist, stats])
        all_feats.append(feat)

    features = np.stack(all_feats)
    log.info("前景顏色特徵 shape: %s（直方圖=%d + 統計量=7）",
             features.shape, bins * 3)
    return features


def extract_size_features(image_paths):
    """提取大小/面積特徵（3 維）。

    計算每張圖中非背景區域的：
    - 面積比例（非黃色像素佔比）
    - Bounding box 長寬比
    - 非背景像素的平均亮度
    """
    log.info("提取大小/面積特徵...")
    all_feats = []
    for p in image_paths:
        img = Image.open(p).convert("HSV")
        arr = np.array(img, dtype=np.uint8)
        h, s, v = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # 非黃色背景的 mask（黃色：H=20~55, S>=40, V>=80）
        not_yellow = ~((h >= 20) & (h <= 55) & (s >= 40) & (v >= 80))
        # 也排除黑色區域
        not_black = v > 30
        foreground = not_yellow & not_black

        total_pixels = h.size
        fg_ratio = foreground.sum() / (total_pixels + 1e-8)

        # Bounding box 長寬比
        rows = np.any(foreground, axis=1)
        cols = np.any(foreground, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_h = rmax - rmin + 1
            bbox_w = cmax - cmin + 1
            aspect_ratio = bbox_w / (bbox_h + 1e-8)
        else:
            aspect_ratio = 1.0

        # 前景平均亮度
        if foreground.any():
            fg_brightness = v[foreground].mean() / 255.0
        else:
            fg_brightness = 0.0

        all_feats.append([fg_ratio, aspect_ratio, fg_brightness])

    features = np.array(all_feats, dtype=np.float32)
    log.info("大小/面積特徵 shape: %s", features.shape)
    return features


def combine_all_features(dino_feat, color_feat, size_feat,
                         color_weight=1.0, size_weight=1.0):
    """合併所有特徵：PCA 自動選維度（保留 95% 變異量）+ UMAP 降維。"""
    from sklearn.decomposition import PCA

    # PCA 降維 DINOv2 特徵，自動選擇維度（保留 95% 變異量）
    log.info("PCA 降維 DINOv2 特徵（%d 維，保留 95%% 變異量）...", dino_feat.shape[1])
    pca_full = PCA(random_state=42)
    pca_full.fit(dino_feat)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumsum, 0.95) + 1)
    n_components = max(n_components, 10)  # 至少保留 10 維
    log.info("  自動選擇 %d 維（保留 %.1f%% 變異量）", n_components, cumsum[n_components - 1] * 100)

    pca = PCA(n_components=n_components, random_state=42)
    dino_pca = pca.fit_transform(dino_feat)

    # L2 normalize 各特徵後加權拼接
    dino_norm = normalize(dino_pca, norm='l2')
    color_norm = normalize(color_feat, norm='l2') * color_weight
    size_norm = normalize(size_feat, norm='l2') * size_weight

    combined = np.concatenate([dino_norm, color_norm, size_norm], axis=1)
    log.info("PCA 合併特徵 shape: %s（DINOv2=%d + 顏色=%d×%.1f + 大小=%d×%.1f）",
             combined.shape, n_components, color_feat.shape[1], color_weight,
             size_feat.shape[1], size_weight)

    # UMAP 降維：保留局部鄰域結構，專為聚類優化
    try:
        import umap
        n_umap = min(30, combined.shape[1] - 1, combined.shape[0] - 2)
        n_umap = max(n_umap, 2)
        log.info("UMAP 降維（%d → %d 維）...", combined.shape[1], n_umap)
        reducer = umap.UMAP(
            n_components=n_umap,
            n_neighbors=min(15, combined.shape[0] - 1),
            min_dist=0.0,       # 聚類用途建議設 0
            metric='cosine',    # cosine 距離更適合 normalize 後的特徵
            random_state=42,
        )
        combined = reducer.fit_transform(combined)
        log.info("UMAP 降維完成，shape: %s", combined.shape)
    except ImportError:
        log.warning("未安裝 umap-learn，跳過 UMAP 降維（pip install umap-learn）")

    return combined


# ============================================================
# 聚類方法
# ============================================================

def run_hdbscan(features, min_cluster_size=15, min_samples=5):
    """執行 HDBSCAN 聚類，噪點自動分配到最近的子群。

    如果 HDBSCAN 找不到任何子群（全部是噪點），會自動降低 min_cluster_size 重試，
    最終 fallback 到 K-Means k=2。
    """
    import hdbscan

    # 嘗試不同的 min_cluster_size，從使用者指定的值開始逐步降低
    attempts = [min_cluster_size]
    for fallback in [30, 20, 10, 5]:
        if fallback < min_cluster_size:
            attempts.append(fallback)

    labels = None
    n_clusters = 0

    for mcs in attempts:
        log.info("執行 HDBSCAN（min_cluster_size=%d, min_samples=%d）...", mcs, min_samples)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min(min_samples, mcs),
            metric='euclidean',  # UMAP 已用 cosine 降維，這裡用 euclidean 即可
        )
        labels = clusterer.fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        log.info("  找到 %d 個子群，%d 個噪點", n_clusters, n_noise)

        if n_clusters >= 2:
            break
        elif n_clusters == 1 and n_noise == 0:
            # 只有 1 群且沒有噪點 = 這個 cluster 本身就很純淨
            log.info("  此 cluster 很純淨，不需要再細分")
            break
        else:
            log.warning("  子群數不足，嘗試降低 min_cluster_size...")

    # 如果所有嘗試都失敗，fallback 到 K-Means
    if n_clusters == 0:
        log.warning("  HDBSCAN 無法找到子群，fallback 到 K-Means k=2")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features)
        n_clusters = 2

    # 把噪點分配到最近的子群
    n_noise = (labels == -1).sum()
    if n_noise > 0 and n_clusters > 0:
        log.info("  將 %d 個噪點分配到最近的子群...", n_noise)
        cluster_ids = sorted(set(labels) - {-1})
        centers = np.array([features[labels == c].mean(axis=0) for c in cluster_ids])

        noise_mask = labels == -1
        noise_features = features[noise_mask]

        from scipy.spatial.distance import cdist
        dists = cdist(noise_features, centers, metric='euclidean')
        nearest = dists.argmin(axis=1)
        labels[noise_mask] = np.array(cluster_ids)[nearest]
        log.info("  噪點已全部分配完畢")

    for c in sorted(set(labels)):
        count = (labels == c).sum()
        log.info("  sub_%d: %d 張", c, count)

    # 計算 Silhouette Score
    score = None
    if len(set(labels)) > 1:
        from sklearn.metrics import silhouette_score
        score = silhouette_score(features, labels)
        log.info("  Silhouette Score: %.4f", score)

    return labels, n_clusters, score


def run_kmeans(features, k):
    """執行 K-Means 聚類。"""
    from sklearn.metrics import silhouette_score

    log.info("執行 K-Means (k=%d)...", k)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)

    score = silhouette_score(features, labels)
    log.info("  Silhouette Score: %.4f", score)

    for c in range(k):
        count = (labels == c).sum()
        log.info("  sub_%d: %d 張", c, count)

    return labels, score


def run_ensemble(features, image_paths, min_cluster_size=15, min_samples=5, n_runs=9):
    """Ensemble Clustering：用不同參數跑多次 HDBSCAN，以共識矩陣決定最終分群。

    流程：
    1. 用不同的 min_cluster_size 和 min_samples 組合跑 n_runs 次 HDBSCAN
    2. 建立共識矩陣（co-association matrix）：兩張圖被分到同一群的次數 / 總次數
    3. 對共識矩陣做 Agglomerative Clustering 得到最終分群
    4. 計算每張圖的「穩定度」：它跟同群其他圖的平均共識分數

    回傳：labels, n_clusters, score, stability（每張圖的穩定度 0~1）
    """
    import hdbscan
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(features)
    log.info("Ensemble Clustering（%d 次 HDBSCAN）...", n_runs)

    # 產生不同的參數組合
    mcs_values = [max(5, min_cluster_size - 10), min_cluster_size, min_cluster_size + 10]
    ms_values = [3, 5, 7]
    param_combos = [(mcs, ms) for mcs in mcs_values for ms in ms_values][:n_runs]

    # 跑多次 HDBSCAN，收集結果
    all_labels = []
    for i, (mcs, ms) in enumerate(param_combos):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min(ms, mcs),
            metric='euclidean',
        )
        labels = clusterer.fit_predict(features)
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        log.info("  第 %d 次（mcs=%d, ms=%d）→ %d 個子群", i + 1, mcs, ms, n_c)
        all_labels.append(labels)

    # 建立共識矩陣
    log.info("建立共識矩陣...")
    coassoc = np.zeros((n, n), dtype=np.float32)
    count = np.zeros((n, n), dtype=np.float32)

    for labels in all_labels:
        for i in range(n):
            for j in range(i + 1, n):
                # 只有兩個都不是噪點時才計入
                if labels[i] >= 0 and labels[j] >= 0:
                    count[i, j] += 1
                    count[j, i] += 1
                    if labels[i] == labels[j]:
                        coassoc[i, j] += 1
                        coassoc[j, i] += 1

    # 避免除以零
    count[count == 0] = 1
    coassoc = coassoc / count
    np.fill_diagonal(coassoc, 1.0)

    # 轉成距離矩陣
    distance = 1.0 - coassoc

    # 用 Agglomerative Clustering 決定最終分群
    # 先用多個 n_clusters 試，選 silhouette 最高的
    best_score = -1
    best_labels = None
    best_k = 2

    # 從多次 HDBSCAN 結果中估計合理的 K 範圍
    k_candidates = set()
    for labels in all_labels:
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        if n_c >= 2:
            k_candidates.add(n_c)
    if not k_candidates:
        k_candidates = {2, 3}
    # 擴展範圍
    k_min = max(2, min(k_candidates) - 1)
    k_max = max(k_candidates) + 1

    log.info("嘗試 K=%d~%d...", k_min, k_max)
    for k in range(k_min, k_max + 1):
        if k >= n:
            continue
        agg = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = agg.fit_predict(distance)
        if len(set(labels)) > 1:
            s = silhouette_score(distance, labels, metric='precomputed')
            log.info("  K=%d → Silhouette=%.4f", k, s)
            if s > best_score:
                best_score = s
                best_labels = labels
                best_k = k

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    log.info("最佳 K=%d（Silhouette=%.4f）", best_k, best_score)

    # 計算每張圖的穩定度
    stability = np.zeros(n)
    for i in range(n):
        same_cluster = best_labels == best_labels[i]
        same_cluster[i] = False
        if same_cluster.sum() > 0:
            stability[i] = coassoc[i, same_cluster].mean()
        else:
            stability[i] = 1.0

    n_stable = (stability >= 0.7).sum()
    n_uncertain = ((stability >= 0.4) & (stability < 0.7)).sum()
    n_unstable = (stability < 0.4).sum()
    log.info("穩定度分佈：穩定=%d, 不確定=%d, 不穩定=%d", n_stable, n_uncertain, n_unstable)

    for c in sorted(set(best_labels)):
        mask = best_labels == c
        avg_stab = stability[mask].mean()
        log.info("  sub_%d: %d 張（平均穩定度=%.2f）", c, mask.sum(), avg_stab)

    return best_labels, best_k, best_score, stability


# ============================================================
# 結果儲存與預覽
# ============================================================

def save_results(image_paths, labels, tag, output_dir: Path, stability=None):
    """將圖片複製到對應的子群資料夾。stability 不為 None 時，不穩定的圖片放到 uncertain/ 資料夾。"""
    out = output_dir / tag
    if out.exists():
        shutil.rmtree(out)

    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        if stability is not None and stability[idx] < 0.4:
            sub_dir = out / "uncertain"
        else:
            name = "noise" if label == -1 else f"sub_{label}"
            sub_dir = out / name
        sub_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, sub_dir / path.name)

    log.info("結果存於 %s", out)
    return out


def make_preview(result_dir: Path, output_dir: Path, cols=10, rows=5, size=128):
    """為每個子群產生預覽圖。"""
    import random

    output_dir.mkdir(parents=True, exist_ok=True)
    subdirs = sorted(p for p in result_dir.iterdir() if p.is_dir())
    n = cols * rows

    for subdir in subdirs:
        tiles = sorted(
            p for p in subdir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        )
        if not tiles:
            continue

        selected = random.sample(tiles, min(n, len(tiles)))
        total_w = cols * size + (cols - 1) * 4
        total_h = rows * size + (rows - 1) * 4
        canvas = Image.new("RGB", (total_w, total_h), (50, 50, 50))

        for idx, path in enumerate(selected):
            r, c = idx // cols, idx % cols
            x, y = c * (size + 4), r * (size + 4)
            try:
                img = Image.open(path).convert("RGB").resize((size, size))
                canvas.paste(img, (x, y))
            except Exception:
                pass

        out_path = output_dir / f"{subdir.name}.jpg"
        canvas.save(out_path, quality=95)
        log.info("  預覽 → %s", out_path)


# ============================================================
# 主流程
# ============================================================

def extract_all_features(image_paths, args):
    """提取並合併所有特徵。"""
    dino_feat = extract_patch_features(image_paths, device=args.device, batch_size=args.batch_size)
    color_feat = extract_color_features(image_paths)
    size_feat = extract_size_features(image_paths)
    features = combine_all_features(
        dino_feat, color_feat, size_feat,
        color_weight=args.color_weight,
        size_weight=args.size_weight,
    )
    return features


def process_one_cluster(cluster_dir, output_dir, args):
    """對單一 cluster 跑子聚類。"""
    log.info("=" * 50)
    log.info("處理 %s", cluster_dir.name)
    log.info("=" * 50)

    image_paths = collect_images(cluster_dir)
    features = extract_all_features(image_paths, args)

    if args.method == 'hdbscan':
        labels, n_clusters, score = run_hdbscan(
            features,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        tag = "hdbscan"
        result_dir = save_results(image_paths, labels, tag, output_dir)

        if args.preview:
            make_preview(result_dir, output_dir / f"{tag}_preview",
                         cols=args.preview_cols, rows=args.preview_rows,
                         size=args.preview_size)

        return {"method": "hdbscan", "n_clusters": n_clusters, "score": score}

    elif args.method == 'ensemble':
        labels, n_clusters, score, stability = run_ensemble(
            features, image_paths,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        tag = "ensemble"
        result_dir = save_results(image_paths, labels, tag, output_dir, stability=stability)

        if args.preview:
            make_preview(result_dir, output_dir / f"{tag}_preview",
                         cols=args.preview_cols, rows=args.preview_rows,
                         size=args.preview_size)

        return {"method": "ensemble", "n_clusters": n_clusters, "score": score}

    else:  # kmeans
        scores = {}
        for k in args.k:
            if k >= len(image_paths):
                log.warning("  K=%d >= 圖片數 %d，跳過", k, len(image_paths))
                continue
            labels, score = run_kmeans(features, k)
            scores[k] = score
            result_dir = save_results(image_paths, labels, f"k{k}", output_dir)

            if args.preview:
                make_preview(result_dir, output_dir / f"k{k}_preview",
                             cols=args.preview_cols, rows=args.preview_rows,
                             size=args.preview_size)

        if len(scores) > 1:
            log.info("-" * 40)
            log.info("%s Silhouette Score 總結：", cluster_dir.name)
            for k, s in sorted(scores.items()):
                marker = " ← 推薦" if s == max(scores.values()) else ""
                log.info("  K=%d : %.4f%s", k, s, marker)

        return {"method": "kmeans", "scores": scores}


def parse_args():
    parser = argparse.ArgumentParser(
        description="對 cluster 進行子聚類（支援 HDBSCAN / K-Means）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="cluster 資料夾，或 --all 時指向包含所有 cluster_X 的父資料夾")
    parser.add_argument("-o", "--output", default=None,
                        help="輸出資料夾（預設：<input>_subcluster）")
    parser.add_argument("--all", action="store_true",
                        help="對 input 底下所有 cluster_X 子資料夾都跑")

    # 聚類方法
    parser.add_argument("--method", choices=["hdbscan", "kmeans", "ensemble"], default="hdbscan",
                        help="聚類方法（ensemble=多次HDBSCAN投票，最穩定）")
    parser.add_argument("-k", "--k", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="K-Means 的 K 值（可多個）")
    parser.add_argument("--min-cluster-size", type=int, default=15,
                        help="HDBSCAN: 最小 cluster 大小")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="HDBSCAN: 核心點最少鄰居數")

    # 特徵權重
    parser.add_argument("--color-weight", type=float, default=1.0,
                        help="顏色特徵權重（0=不用, 1=等權重, 3=顏色主導）")
    parser.add_argument("--size-weight", type=float, default=1.0,
                        help="大小/面積特徵權重（0=不用, 1=等權重）")

    # 裝置
    parser.add_argument("--device", default="cuda", help="計算裝置")
    parser.add_argument("--batch-size", type=int, default=32, help="特徵提取 batch size")

    # 預覽
    parser.add_argument("--preview", action="store_true", help="產生預覽圖")
    parser.add_argument("--preview-cols", type=int, default=10, help="預覽圖每排張數")
    parser.add_argument("--preview-rows", type=int, default=5, help="預覽圖排數")
    parser.add_argument("--preview-size", type=int, default=128, help="預覽圖每張大小")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input)

    if args.all:
        cluster_dirs = sorted(
            p for p in input_dir.iterdir()
            if p.is_dir() and p.name.startswith("cluster_")
        )
        if not cluster_dirs:
            log.error("在 '%s' 找不到任何 cluster_X 資料夾", input_dir)
            exit(1)

        log.info("找到 %d 個 cluster，方法：%s", len(cluster_dirs), args.method)
        all_results = {}

        for cluster_dir in cluster_dirs:
            output_dir = input_dir / f"{cluster_dir.name}_subcluster"
            result = process_one_cluster(cluster_dir, output_dir, args)
            all_results[cluster_dir.name] = result

        log.info("\n" + "=" * 60)
        log.info("全部 cluster 子聚類結果總結（%s）", args.method)
        log.info("=" * 60)
        for name, result in sorted(all_results.items()):
            if result["method"] in ("hdbscan", "ensemble"):
                s = f"score={result['score']:.4f}" if result['score'] else "N/A"
                log.info("  %s → %d 個子群 (%s)", name, result["n_clusters"], s)
            else:
                scores = result["scores"]
                if scores:
                    best_k = max(scores, key=scores.get)
                    log.info("  %s → 建議 K=%d (score=%.4f)", name, best_k, scores[best_k])
    else:
        output_dir = Path(args.output) if args.output else input_dir.parent / f"{input_dir.name}_subcluster"
        process_one_cluster(input_dir, output_dir, args)

    log.info("✅ 全部完成！")
