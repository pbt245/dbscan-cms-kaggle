import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from hdbscan import HDBSCAN
import hdbscan  # Need this for approximate_predict
import wandb

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits, get_datasets_with_gcdval
from project_utils.cluster_and_log_utils import log_accs_from_preds
from models.dino import DINO


def reassign_noise_to_clusters(preds, feats_np, clustering, method="approximate"):
    """
    Reassign noise points (-1) to nearest clusters

    Args:
        preds: Cluster predictions (contains -1 for noise)
        feats_np: Normalized feature array
        clustering: Fitted HDBSCAN object
        method: "approximate" (uses HDBSCAN tree) or "centroid" (distance-based)

    Returns:
        preds: Updated predictions with noise reassigned
        n_noise: Number of noise points reassigned
        method_used: Which method was actually used
    """
    n_noise = list(preds).count(-1)

    if n_noise == 0:
        return preds, 0, "none"

    noise_mask = preds == -1
    valid_mask = ~noise_mask

    if valid_mask.sum() == 0:
        print("  All points are noise! Cannot reassign.")
        return preds, n_noise, "failed"

    # Method 1: Try HDBSCAN's approximate_predict
    if method == "approximate" and hasattr(clustering, "prediction_data_"):
        try:
            noise_indices = np.where(noise_mask)[0]
            noise_features = feats_np[noise_mask]

            # Use HDBSCAN's built-in prediction
            approximate_labels, strengths = hdbscan.approximate_predict(
                clustering, noise_features
            )

            # Assign predictions to noise points
            preds[noise_indices] = approximate_labels

            avg_confidence = strengths.mean()
            print(
                f"Used HDBSCAN approximate_predict (avg confidence: {avg_confidence:.3f})"
            )
            return preds, n_noise, "approximate"

        except Exception as e:
            print(f"  Approximate predict failed: {e}")
            print(f" Falling back to centroid-based assignment")

    # Method 2: Fallback - Centroid-based assignment
    unique_labels = np.unique(preds[valid_mask])

    # Calculate centroids once (efficient!)
    centroids = np.array(
        [feats_np[preds == label].mean(axis=0) for label in unique_labels]
    )

    # Vectorized distance computation for all noise points at once
    noise_points = feats_np[noise_mask]
    distances = np.linalg.norm(
        noise_points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
    )
    nearest_clusters = unique_labels[np.argmin(distances, axis=1)]
    preds[noise_mask] = nearest_clusters

    print(f"Used centroid-based assignment")
    return preds, n_noise, "centroid"


def remap_cluster_labels(preds):
    """
    Remap cluster labels to be contiguous (0, 1, 2, ..., k-1)

    Args:
        preds: Cluster predictions (may have gaps like 0, 2, 5, 7)

    Returns:
        preds: Remapped predictions (0, 1, 2, 3)
        n_clusters: Number of unique clusters
    """
    unique_labels = np.unique(preds)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    preds = np.array([label_map[p] for p in preds])
    return preds, len(unique_labels)


def iterative_meanshift_hdbscan(model, loader, args):
    """
    HDBSCAN clustering with iterative mean-shift feature refinement
    HDBSCAN is hierarchical and doesn't require eps parameter
    """
    print("\n" + "=" * 80)
    print(f"HDBSCAN Configuration:")
    print(f"  min_cluster_size: {args.min_cluster_size}")
    print(f"  min_samples: {args.min_samples}")
    print(f"  Mean-shift k: {args.k}")
    print(f"  Mean-shift alpha: {args.alpha}")
    print("=" * 80 + "\n")

    # Initialize wandb run
    if args.wandb:
        wandb.init(
            project="hdbscan-test",
            name=f"{args.dataset_name}_mcs{args.min_cluster_size}_ms{args.min_samples}",
            config={
                "dataset": args.dataset_name,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "k": args.k,
                "alpha": args.alpha,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        )

    all_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim))
    new_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim))
    targets = torch.zeros(len(loader.dataset), dtype=int)
    mask_lab = torch.zeros(len(loader.dataset), dtype=bool)
    mask_cls = torch.zeros(len(loader.dataset), dtype=bool)

    best_acc = 0
    best_result = (0, 0, 0)
    tolerance_count = 0
    max_tolerance = 3

    with torch.no_grad():
        for epoch in range(args.epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*80}")

            # ====================
            # 1. FEATURE EXTRACTION / REFINEMENT
            # ====================
            for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting features")):
                images, label, uq_idxs, mask_lab_ = batch

                if epoch == 0:
                    # Initial feature extraction
                    images = torch.Tensor(images).to(device)
                    feats = model(images).detach().cpu()
                    all_feats[uq_idxs] = feats
                    targets[uq_idxs] = label.cpu()
                    mask_lab[uq_idxs] = mask_lab_.squeeze(1).cpu().bool()
                else:
                    # Mean-shift refinement
                    classwise_sim = torch.einsum(
                        "b d, n d -> b n", all_feats[uq_idxs], all_feats
                    )
                    _, indices = classwise_sim.topk(
                        k=args.k + 1, dim=-1, largest=True, sorted=True
                    )
                    indices = indices[:, 1:]  # Exclude self
                    knn_emb = torch.mean(
                        all_feats[indices].view(-1, args.k, args.feat_dim), dim=1
                    )
                    new_feats[uq_idxs] = (1 - args.alpha) * all_feats[
                        uq_idxs
                    ] + args.alpha * knn_emb.detach().cpu()

            # ====================
            # 2. SETUP MASKS & NORMALIZE
            # ====================
            if epoch == 0:
                mask_cls = np.isin(targets, range(len(args.train_classes))).astype(bool)
                mask_lab = mask_lab.numpy().astype(bool)
                l_targets = targets[mask_lab].numpy()
                u_targets = targets[~mask_lab].numpy()
                mask = mask_cls[~mask_lab].astype(bool)
            else:
                # Normalize features (CRITICAL for cosine similarity!)
                norm = torch.sqrt(
                    torch.sum(torch.pow(new_feats, 2), dim=-1, keepdim=True)
                )
                new_feats = new_feats / (norm + 1e-8)
                all_feats = new_feats

            # ====================
            # 3. HDBSCAN CLUSTERING
            # ====================
            print("\n Running HDBSCAN clustering...")

            # Convert to numpy and ensure normalization
            feats_np = all_feats.numpy()
            norm = np.linalg.norm(feats_np, axis=1, keepdims=True)
            feats_np = feats_np / (norm + 1e-8)

            # Run HDBSCAN
            clustering = HDBSCAN(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                metric="euclidean",  # On normalized vectors = cosine similarity
                cluster_selection_method="eom",  # Excess of Mass (more stable)
                core_dist_n_jobs=-1,  # Use all CPU cores
            )

            preds = clustering.fit_predict(feats_np)

            # Get clustering statistics
            n_clusters_raw = len(set(preds)) - (1 if -1 in preds else 0)
            n_noise = list(preds).count(-1)

            print(
                f" Raw results: {n_clusters_raw} clusters, {n_noise} noise points"
            )

            # ====================
            # 4. HANDLE NOISE POINTS
            # ====================
            if n_noise > 0:
                print(f"\n Reassigning {n_noise} noise points...")
                preds, n_noise, method_used = reassign_noise_to_clusters(
                    preds, feats_np, clustering, method="approximate"
                )

            # ====================
            # 5. REMAP LABELS
            # ====================
            preds, n_clusters_final = remap_cluster_labels(preds)
            print(f"   Final: {n_clusters_final} clusters")

            if n_clusters_final == 0:
                print(
                    "\nNo clusters found! Try decreasing min_cluster_size or min_samples"
                )
                break

            # ====================
            # 6. EVALUATION
            # ====================
            from project_utils.cluster_utils import cluster_acc

            # Labeled train accuracy (stopping criterion)
            old_acc_train = cluster_acc(l_targets, preds[mask_lab])
            print(f"\nLabeled train accuracy: {old_acc_train:.4f}")

            # Unlabeled train accuracy (main metric)
            all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(
                y_true=u_targets,
                y_pred=preds[~mask_lab],
                mask=mask,
                T=epoch,
                eval_funcs=args.eval_funcs,
                save_name="HDBSCAN",
                print_output=True,
            )

            # ====================
            # 7. WANDB LOGGING
            # ====================
            if args.wandb:
                wandb.log(
                    {
                        # Main metrics
                        "epoch": epoch,
                        "unlabeled/all_acc": all_acc_test,
                        "unlabeled/old_acc": old_acc_test,
                        "unlabeled/new_acc": new_acc_test,
                        "labeled/train_acc": old_acc_train,
                        # Clustering quality
                        "clustering/n_clusters": n_clusters_final,
                        "clustering/n_noise_points": n_noise,
                        "clustering/noise_ratio": n_noise / len(preds),
                        # Best metrics tracking
                        "best/labeled_acc": best_acc,
                        "best/unlabeled_all": best_result[0],
                        "best/unlabeled_old": best_result[1],
                        "best/unlabeled_new": best_result[2],
                        # Early stopping
                        "early_stopping/tolerance_count": tolerance_count,
                    }
                )

            # ====================
            # 8. EARLY STOPPING
            # ====================
            if old_acc_train > best_acc:
                best_acc = old_acc_train
                best_result = (all_acc_test, old_acc_test, new_acc_test)
                tolerance_count = 0
                print(f"New best labeled accuracy: {best_acc:.4f}")
            else:
                tolerance_count += 1
                print(
                    f"No improvement. Tolerance: {tolerance_count}/{max_tolerance}"
                )

            if tolerance_count >= max_tolerance:
                print("\nEarly stopping due to no improvement")
                break

    # ====================
    # FINAL RESULTS
    # ====================
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best labeled train accuracy: {best_acc:.4f}")
    print(f"Unlabeled test accuracies:")
    print(f"  All:   {best_result[0]:.4f}")
    print(f"  Old:   {best_result[1]:.4f}")
    print(f"  New:   {best_result[2]:.4f}")
    print(f"{'='*80}\n")

    if args.wandb:
        wandb.log(
            {
                "final/best_labeled_acc": best_acc,
                "final/best_unlabeled_all": best_result[0],
                "final/best_unlabeled_old": best_result[1],
                "final/best_unlabeled_new": best_result[2],
            }
        )
        wandb.finish()

    return best_result


def iterative_meanshift_hdbscan_inductive(model, loader, val_loader, args):
    """
    HDBSCAN clustering for inductive setting (with validation set)
    """
    print("\n" + "=" * 80)
    print(f"HDBSCAN Configuration (Inductive):")
    print(f"  min_cluster_size: {args.min_cluster_size}")
    print(f"  min_samples: {args.min_samples}")
    print("=" * 80 + "\n")

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="CMS-HDBSCAN-Inductive",
            name=f"{args.dataset_name}_mcs{args.min_cluster_size}_ms{args.min_samples}",
            config={
                "dataset": args.dataset_name,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "k": args.k,
                "alpha": args.alpha,
                "epochs": args.epochs,
                "inductive": True,
            },
        )

    all_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim))
    new_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim))
    targets = torch.zeros(len(loader.dataset), dtype=int)
    mask_cls = torch.zeros(len(loader.dataset), dtype=bool)

    all_feats_val = []
    new_feats_val = []
    targets_val = []
    mask_cls_val = []

    best_acc = 0
    best_result = (0, 0, 0)
    tolerance_count = 0
    max_tolerance = 3

    with torch.no_grad():
        for epoch in range(args.epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*80}")

            # Process test data
            for batch_idx, batch in enumerate(tqdm(loader, desc="Test features")):
                images, label, uq_idxs = batch
                if epoch == 0:
                    images = torch.Tensor(images).to(device)
                    all_feats[uq_idxs] = model(images).detach().cpu()
                    targets[uq_idxs] = label.cpu()
                else:
                    classwise_sim = torch.einsum(
                        "b d, n d -> b n", all_feats[uq_idxs], all_feats
                    )
                    _, indices = classwise_sim.topk(
                        k=args.k + 1, dim=-1, largest=True, sorted=True
                    )
                    indices = indices[:, 1:]
                    knn_emb = torch.mean(
                        all_feats[indices].view(-1, args.k, args.feat_dim), dim=1
                    )
                    new_feats[uq_idxs] = (1 - args.alpha) * all_feats[
                        uq_idxs
                    ] + args.alpha * knn_emb.detach().cpu()

            if epoch == 0:
                mask_cls = np.isin(targets, range(len(args.train_classes))).astype(bool)
                targets = np.array(targets)
            else:
                norm = torch.sqrt(
                    torch.sum(torch.pow(new_feats, 2), dim=-1, keepdim=True)
                )
                new_feats = new_feats / (norm + 1e-8)
                all_feats = new_feats

            # Process validation data
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val features")):
                images, label, uq_idxs = batch[:3]
                if epoch == 0:
                    images = torch.Tensor(images).to(device)
                    all_feats_val.append(model(images).detach().cpu())
                    targets_val.append(label.cpu())
                else:
                    start_idx = batch_idx * args.batch_size
                    end_idx = start_idx + len(uq_idxs)
                    classwise_sim = torch.einsum(
                        "b d, n d -> b n",
                        torch.cat(all_feats_val)[start_idx:end_idx],
                        torch.cat(all_feats_val),
                    )
                    _, indices = classwise_sim.topk(
                        k=args.k + 1, dim=-1, largest=True, sorted=True
                    )
                    indices = indices[:, 1:]
                    knn_emb_val = torch.mean(
                        torch.cat(all_feats_val)[indices].view(
                            -1, args.k, args.feat_dim
                        ),
                        dim=1,
                    )
                    new_feats_val[start_idx:end_idx] = (1 - args.alpha) * torch.cat(
                        all_feats_val
                    )[start_idx:end_idx] + args.alpha * knn_emb_val.detach().cpu()

            if epoch == 0:
                all_feats_val = torch.cat(all_feats_val)
                targets_val = np.array(torch.cat(targets_val))
                mask_cls_val = np.isin(
                    targets_val, range(len(args.train_classes))
                ).astype(bool)
                new_feats_val = all_feats_val
            else:
                norm = torch.sqrt(
                    torch.sum(torch.pow(new_feats_val, 2), dim=-1, keepdim=True)
                )
                new_feats_val = new_feats_val / (norm + 1e-8)
                all_feats_val = new_feats_val

            # HDBSCAN on test and validation
            print("\n Clustering test data...")
            feats_test_np = all_feats.numpy()
            norm = np.linalg.norm(feats_test_np, axis=1, keepdims=True)
            feats_test_np = feats_test_np / (norm + 1e-8)

            print(" Clustering validation data...")
            feats_val_np = all_feats_val.numpy()
            norm = np.linalg.norm(feats_val_np, axis=1, keepdims=True)
            feats_val_np = feats_val_np / (norm + 1e-8)

            # Cluster test data
            clustering_test = HDBSCAN(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                core_dist_n_jobs=-1,
            )
            preds_test = clustering_test.fit_predict(feats_test_np)

            # Cluster validation data
            clustering_val = HDBSCAN(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                core_dist_n_jobs=-1,
            )
            preds_val = clustering_val.fit_predict(feats_val_np)

            # Handle noise points for both
            n_noise_test = list(preds_test).count(-1)
            n_noise_val = list(preds_val).count(-1)

            print(
                f"Test: {len(set(preds_test)) - (1 if -1 in preds_test else 0)} clusters, {n_noise_test} noise"
            )
            print(
                f"Val:  {len(set(preds_val)) - (1 if -1 in preds_val else 0)} clusters, {n_noise_val} noise"
            )

            if n_noise_test > 0:
                preds_test, _, _ = reassign_noise_to_clusters(
                    preds_test, feats_test_np, clustering_test
                )

            if n_noise_val > 0:
                preds_val, _, _ = reassign_noise_to_clusters(
                    preds_val, feats_val_np, clustering_val
                )

            # Remap labels
            preds_test, n_clusters_test = remap_cluster_labels(preds_test)
            preds_val, n_clusters_val = remap_cluster_labels(preds_val)

            print(
                f"Final - Test: {n_clusters_test} clusters, Val: {n_clusters_val} clusters"
            )

            # Evaluate
            from project_utils.cluster_utils import cluster_acc

            old_acc_val = cluster_acc(
                targets_val[mask_cls_val], preds_val[mask_cls_val]
            )
            print(f"\nVal labeled accuracy: {old_acc_val:.4f}")

            all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(
                y_true=targets,
                y_pred=preds_test,
                mask=mask_cls,
                T=epoch,
                eval_funcs=args.eval_funcs,
                save_name="HDBSCAN test",
                print_output=True,
            )

            # Wandb logging
            if args.wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "test/all_acc": all_acc_test,
                        "test/old_acc": old_acc_test,
                        "test/new_acc": new_acc_test,
                        "val/labeled_acc": old_acc_val,
                        "clustering/test_n_clusters": n_clusters_test,
                        "clustering/val_n_clusters": n_clusters_val,
                        "clustering/test_noise": n_noise_test,
                        "clustering/val_noise": n_noise_val,
                    }
                )

            if old_acc_val > best_acc:
                best_acc = old_acc_val
                best_result = (all_acc_test, old_acc_test, new_acc_test)
                tolerance_count = 0
                print(f"New best val accuracy: {best_acc:.4f}")
            else:
                tolerance_count += 1
                print(
                    f"No improvement. Tolerance: {tolerance_count}/{max_tolerance}"
                )

            if tolerance_count >= max_tolerance:
                print("\nEarly stopping")
                break

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (Inductive)")
    print(f"{'='*80}")
    print(f"Best val labeled accuracy: {best_acc:.4f}")
    print(f"Test accuracies:")
    print(f"  All: {best_result[0]:.4f}")
    print(f"  Old: {best_result[1]:.4f}")
    print(f"  New: {best_result[2]:.4f}")
    print(f"{'='*80}\n")

    if args.wandb:
        wandb.log(
            {
                "final/best_val_acc": best_acc,
                "final/test_all": best_result[0],
                "final/test_old": best_result[1],
                "final/test_new": best_result[2],
            }
        )
        wandb.finish()

    return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDBSCAN clustering with mean-shift")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int) # kaggle max =4
    parser.add_argument("--model_name", type=str, default="cifar100_best")
    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--k", default=8, type=int, help="K for mean-shift")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mean-shift alpha")
    parser.add_argument("--pretrain_path", type=str, default="./models")

    # HDBSCAN specific parameters
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="HDBSCAN: minimum size of clusters (5-50 typical)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="HDBSCAN: how conservative clustering is (1-10 typical)",
    )

    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--eval_funcs", nargs="+", default=["v2"])
    parser.add_argument("--use_ssb_splits", type=bool, default=True)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.feat_dim = 768
    args.interpolation = 3
    args.crop_pct = 0.875
    args.prop_train_labels = 0.5
    args.num_mlp_layers = 3

    print("\n" + "=" * 80)
    print("HDBSCAN CLUSTERING CONFIGURATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"Min samples: {args.min_samples}")
    print(f"Inductive: {args.inductive}")
    print(f"Wandb: {args.wandb}")
    print("=" * 80 + "\n")

    # Load model
    model_path = (
        f"./log/metric_learn_gcd/log/{args.model_name}/checkpoints/model_best.pt"
    )
    print(f"Loading model from: {model_path}")

    model = DINO(args)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict["model_state_dict"], strict=False)

    # Load data
    from data.augmentations import get_transform

    train_transform, test_transform = get_transform(
        "imagenet", image_size=224, args=args
    )

    if args.inductive:
        _, test_dataset, _, val_dataset, _ = get_datasets_with_gcdval(
            args.dataset_name, test_transform, test_transform, args
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        iterative_meanshift_hdbscan_inductive(model, test_loader, val_loader, args)
    else:
        train_dataset, _, _, _ = get_datasets(
            args.dataset_name, test_transform, test_transform, args
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        iterative_meanshift_hdbscan(model, train_loader, args)
