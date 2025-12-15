import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from hdbscan import HDBSCAN

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits, get_datasets_with_gcdval
from project_utils.cluster_and_log_utils import log_accs_from_preds
from models.dino import DINO


def iterative_meanshift_hdbscan(model, loader, args):
    """
    HDBSCAN clustering with iterative mean-shift feature refinement
    HDBSCAN is hierarchical and doesn't require eps parameter
    """
    print(
        f"Using HDBSCAN with min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}"
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
            print(f"\n=== Epoch {epoch} ===")

            # Extract or refine features
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

            if epoch == 0:
                mask_cls = np.isin(targets, range(len(args.train_classes))).astype(bool)
                mask_lab = mask_lab.numpy().astype(bool)
                l_targets = targets[mask_lab].numpy()
                u_targets = targets[~mask_lab].numpy()
                mask = mask_cls[~mask_lab].astype(bool)
            else:
                # Normalize features (CRITICAL!)
                norm = torch.sqrt(torch.sum(torch.pow(new_feats, 2), dim=-1, keepdim=True))
                new_feats = new_feats / (norm + 1e-8)
                all_feats = new_feats

            # === HDBSCAN Clustering ===
            print("Running HDBSCAN clustering...")

            # Convert to numpy and ensure normalization
            feats_np = all_feats.numpy()
            norm = np.linalg.norm(feats_np, axis=1, keepdims=True)
            feats_np = feats_np / (norm + 1e-8)

            # Run HDBSCAN
            # HDBSCAN parameters:
            # - min_cluster_size: minimum size of clusters
            # - min_samples: how conservative the clustering is (higher = more conservative)
            # - metric: distance metric
            # - cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'
            clustering = HDBSCAN(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                metric="euclidean",  # On normalized vectors
                cluster_selection_method="eom",  # More stable than 'leaf'
                core_dist_n_jobs=-1,
            )

            preds = clustering.fit_predict(feats_np)

            # Get clustering statistics
            n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
            n_noise = list(preds).count(-1)

            print(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

            if n_clusters == 0:
                print(
                    "No clusters found! Try decreasing min_cluster_size or min_samples"
                )
                break

            # Reassign noise points to nearest cluster using cluster probabilities if available
            if n_noise > 0:
                print(f"Reassigning {n_noise} noise points to nearest clusters...")
                noise_mask = preds == -1
                valid_mask = ~noise_mask

                if valid_mask.sum() > 0:
                    # Use HDBSCAN's membership probability if available
                    if hasattr(clustering, "probabilities_"):
                        probs = clustering.probabilities_
                        # For noise points (prob = 0), assign to nearest cluster
                        for i in np.where(noise_mask)[0]:
                            unique_labels = np.unique(preds[valid_mask])
                            centroids = np.array(
                                [
                                    feats_np[preds == label].mean(axis=0)
                                    for label in unique_labels
                                ]
                            )
                            distances = np.linalg.norm(centroids - feats_np[i], axis=1)
                            preds[i] = unique_labels[np.argmin(distances)]
                    else:
                        # Fallback to centroid-based assignment
                        unique_labels = np.unique(preds[valid_mask])
                        centroids = np.array(
                            [
                                feats_np[preds == label].mean(axis=0)
                                for label in unique_labels
                            ]
                        )

                        for i in np.where(noise_mask)[0]:
                            distances = np.linalg.norm(centroids - feats_np[i], axis=1)
                            preds[i] = unique_labels[np.argmin(distances)]

            # Remap labels to be contiguous (0, 1, 2, ...)
            unique_labels = np.unique(preds)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            preds = np.array([label_map[p] for p in preds])

            print(f"Final: {len(unique_labels)} clusters")

            # Evaluate on labeled train data (stopping criterion)
            from project_utils.cluster_utils import cluster_acc

            old_acc_train = cluster_acc(l_targets, preds[mask_lab])
            print(f"Labeled train accuracy: {old_acc_train:.4f}")

            # Evaluate on unlabeled train data
            all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(
                y_true=u_targets,
                y_pred=preds[~mask_lab],
                mask=mask,
                T=epoch,
                eval_funcs=args.eval_funcs,
                save_name="HDBSCAN unlabeled train ACC",
                print_output=True,
            )

            # Stopping criterion with tolerance
            if old_acc_train > best_acc:
                best_acc = old_acc_train
                best_result = (all_acc_test, old_acc_test, new_acc_test)
                tolerance_count = 0
                print(f"New best labeled accuracy: {best_acc:.4f}")
            else:
                tolerance_count += 1
                print(f"No improvement. Tolerance: {tolerance_count}/{max_tolerance}")

            if tolerance_count >= max_tolerance:
                print("Early stopping due to no improvement")
                break

    print(f"\n=== Final Results ===")
    print(f"Best labeled train accuracy: {best_acc:.4f}")
    print(
        f"Unlabeled test - All: {best_result[0]:.4f} | Old: {best_result[1]:.4f} | New: {best_result[2]:.4f}"
    )

    return best_result


def iterative_meanshift_hdbscan_inductive(model, loader, val_loader, args):
    """
    HDBSCAN clustering for inductive setting
    """
    print(
        f"Using HDBSCAN with min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}"
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
            print(f"\n=== Epoch {epoch} ===")

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
            feats_test_np = all_feats.numpy()
            norm = np.linalg.norm(feats_test_np, axis=1, keepdims=True)
            feats_test_np = feats_test_np / (norm + 1e-8)

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
            for preds, feats_np, name in [
                (preds_test, feats_test_np, "test"),
                (preds_val, feats_val_np, "val"),
            ]:
                n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
                n_noise = list(preds).count(-1)

                if n_noise > 0:
                    noise_mask = preds == -1
                    valid_mask = ~noise_mask
                    if valid_mask.sum() > 0:
                        unique_labels = np.unique(preds[valid_mask])
                        centroids = np.array(
                            [
                                feats_np[preds == label].mean(axis=0)
                                for label in unique_labels
                            ]
                        )
                        for i in np.where(noise_mask)[0]:
                            distances = np.linalg.norm(centroids - feats_np[i], axis=1)
                            preds[i] = unique_labels[np.argmin(distances)]

                # Remap labels
                unique_labels = np.unique(preds)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                preds[:] = [label_map[p] for p in preds]
                print(f"{name}: {n_clusters} clusters, {n_noise} noise reassigned")

            # Evaluate
            from project_utils.cluster_utils import cluster_acc

            old_acc_val = cluster_acc(
                targets_val[mask_cls_val], preds_val[mask_cls_val]
            )
            print(f"Val labeled accuracy: {old_acc_val:.4f}")

            all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(
                y_true=targets,
                y_pred=preds_test,
                mask=mask_cls,
                T=epoch,
                eval_funcs=args.eval_funcs,
                save_name="HDBSCAN test ACC",
                print_output=True,
            )

            if old_acc_val > best_acc:
                best_acc = old_acc_val
                best_result = (all_acc_test, old_acc_test, new_acc_test)
                tolerance_count = 0
            else:
                tolerance_count += 1

            if tolerance_count >= max_tolerance:
                break

    print(f"\n=== Final Results ===")
    print(f"Best val labeled accuracy: {best_acc:.4f}")
    print(
        f"Test - All: {best_result[0]:.4f} | Old: {best_result[1]:.4f} | New: {best_result[2]:.4f}"
    )

    return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDBSCAN clustering")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--model_name", type=str, default="cifar100_best")
    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--pretrain_path", type=str, default="/kaggle/working/dbscan-for-cms/models"
    )

    # HDBSCAN specific parameters
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="HDBSCAN min_cluster_size parameter (minimum size of clusters)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="HDBSCAN min_samples parameter (how conservative clustering is)",
    )

    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--eval_funcs", nargs="+", default=["v2"])
    parser.add_argument("--use_ssb_splits", type=bool, default=True)

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

    print(args)

    # Load model
    model_path = (
        f"./log/metric_learn_gcd/log/{args.model_name}/checkpoints/model_best.pt"
    )
    print(f"Loading model from {model_path}")

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
