import argparse
import os

import torch
import numpy as np
import torch.nn as nn
import wandb

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from project_utils.cluster_utils import AverageMeter
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_datasets_with_gcdval, get_class_splits
from project_utils.cluster_and_log_utils import *
from project_utils.general_utils import init_experiment, str2bool

from models.dino import *
from methods.loss import *
from config import CMS_ROOT

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

wandb.login(key="b264a66ddb02e3aab0297a18e30ce8cd996dc863")


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class ArcFaceHead(nn.Module):
    """ArcFace classification head for labeled data only"""

    def __init__(self, in_dim, num_classes, s=64.0, m=0.5):
        super(ArcFaceHead, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m

        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m) for ArcFace
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: normalized features [batch_size, in_dim]
            labels: ground truth labels [batch_size]
        Returns:
            logits: scaled logits with ArcFace margin [batch_size, num_classes]
        """
        # Normalize weights
        weight_norm = F.normalize(self.weight, dim=1)

        # Compute cosine similarity
        cos_theta = F.linear(embeddings, weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)

        # Compute sin(theta)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        # Compute cos(theta + m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Apply the margin only to the target class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Use cos(theta + m) for target class, cos(theta) for others
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)

        # Scale by s
        output = output * self.s

        return output


class DINO_with_ArcFace(nn.Module):
    """DINO model with ArcFace head for labeled data"""

    def __init__(self, args):
        super(DINO_with_ArcFace, self).__init__()
        self.k = args.k
        self.backbone = self.init_backbone(args.pretrain_path)
        self.img_projection_head = vits.__dict__["DINOHead"](
            in_dim=args.feat_dim, out_dim=args.feat_dim, nlayers=args.num_mlp_layers
        )

        # ArcFace head for labeled data classification
        self.arcface_head = ArcFaceHead(
            in_dim=args.feat_dim,
            num_classes=args.num_labeled_classes,
            s=args.arcface_s,
            m=args.arcface_m,
        )

    def forward(self, image, return_features_only=False):
        """
        Args:
            image: input images
            return_features_only: if True, only return normalized features
        Returns:
            features: normalized projection head output (v_i in paper)
        """
        feat = self.backbone(image)
        feat = self.img_projection_head(feat)
        feat = F.normalize(feat, dim=-1)

        return feat

    def forward_arcface(self, features, labels):
        """
        Forward pass through ArcFace head
        Args:
            features: normalized features from projection head
            labels: ground truth labels for labeled data
        Returns:
            logits: ArcFace logits
        """
        return self.arcface_head(features, labels)

    def init_backbone(self, pretrain_path):
        model = vits.__dict__["vit_base"]()
        state_dict = torch.load(
            os.path.join(pretrain_path, "dino_vitbase16_pretrain.pth"),
            map_location="cpu",
        )
        model.load_state_dict(state_dict)
        for m in model.parameters():
            m.requires_grad = False

        for name, m in model.named_parameters():
            if "block" in name:
                block_num = int(name.split(".")[1])
                if block_num >= 11:
                    m.requires_grad = True

        return model


def train(model, train_loader, test_loader, train_loader_memory, args):

    # Optimizer includes ArcFace head parameters
    optimizer = SGD(
        list(model.module.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * args.eta_min,
    )

    sup_con_crit = SupConLoss()
    unsup_con_crit = ConMeanShiftLoss(args)
    ce_criterion = nn.CrossEntropyLoss()

    best_agglo_score = 0
    best_agglo_img_score = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        loss_cms_unsup_record = AverageMeter()
        loss_cms_sup_record = AverageMeter()
        loss_arc_record = AverageMeter()

        # ==========================================
        # Step 1: Build memory bank of all features
        # ==========================================
        with torch.no_grad():
            model.eval()
            all_feats = []
            for batch_idx, batch in enumerate(
                tqdm(train_loader_memory, desc="Building memory bank")
            ):
                images, class_labels, uq_idxs, mask_lab = batch
                images = torch.cat(images, dim=0).to(device)

                features = model(images)  # Get v_i (normalized projection output)
                all_feats.append(features.detach().cpu())
            all_feats = torch.cat(all_feats)

        # ==========================================
        # Step 2: Training loop
        # ==========================================
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images, class_labels, uq_idxs, mask_lab = batch
            class_labels, mask_lab = (
                class_labels.to(device),
                mask_lab[:, 0].to(device).bool(),
            )
            images = torch.cat(images, dim=0).to(device)

            # Forward: Get normalized features v_i
            features = model(images)

            # ==========================================
            # Step 3: Mean-Shift (CMS)
            # ==========================================
            # Compute kNN from memory bank
            # if batch_idx == 0:
            #     all_feats_gpu = all_feats.to(device)
            classwise_sim = torch.einsum(
                "b d, n d -> b n", features.cpu(), all_feats
            )  # might be upgrade by remove .cpu()
            _, indices = classwise_sim.topk(
                k=args.k + 1, dim=-1, largest=True, sorted=True
            )
            indices = indices[:, 1:]  # Exclude self
            knn_emb = torch.mean(
                all_feats[indices, :].view(-1, args.k, args.feat_dim), dim=1
            ).to(device)

            # ==========================================
            # Step 4: CMS Losses
            # ==========================================
            if args.contrast_unlabel_only:
                # Unsupervised contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
                f3, f4 = [f[~mask_lab] for f in knn_emb.chunk(2)]
                con_knn_emb = torch.cat([f3, f4], dim=0)
            else:
                # Unsupervised contrastive loss for all examples
                con_feats = features
                con_knn_emb = knn_emb

            unsup_con_loss = unsup_con_crit(con_knn_emb, con_feats)

            # Supervised contrastive loss (on labeled data)
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            if mask_lab.sum() > 0:
                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            else:
                sup_con_loss = torch.tensor(0.0).to(device)

            # ==========================================
            # Step 5: ArcFace Loss (only on labeled data)
            # ==========================================
            if mask_lab.sum() > 0:
                # get features for labeled data (first view only)
                labeled_features = features.chunk(2)[0][
                    mask_lab
                ]  # v_i for labeled samples
                labeled_targets = class_labels[mask_lab]

                # forward through ArcFace head
                arcface_logits = model.module.forward_arcface(
                    labeled_features, labeled_targets
                )
                arcface_loss = ce_criterion(arcface_logits, labeled_targets)
            else:
                arcface_loss = torch.tensor(0.0).to(device)

            # ==========================================
            # Step 6: Total Loss
            # ==========================================
            loss = (
                (1 - args.sup_con_weight) * unsup_con_loss
                + args.sup_con_weight * sup_con_loss
                + args.arcface_weight * arcface_loss
            )

            # save losses
            loss_record.update(loss.item(), class_labels.size(0))
            loss_cms_unsup_record.update(unsup_con_loss.item(), class_labels.size(0))
            loss_cms_sup_record.update(
                sup_con_loss.item() if isinstance(sup_con_loss, torch.Tensor) else 0.0,
                class_labels.size(0),
            )
            loss_arc_record.update(
                arcface_loss.item() if isinstance(arcface_loss, torch.Tensor) else 0.0,
                class_labels.size(0),
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Train Epoch: {epoch} | Total Loss: {loss_record.avg:.4f} | "
            f"CMS Unsup: {loss_cms_unsup_record.avg:.4f} | "
            f"CMS Sup: {loss_cms_sup_record.avg:.4f} | "
            f"ArcFace: {loss_arc_record.avg:.4f}"
        )

        # ==========================================
        # Step 7: Evaluation (Same as original CMS)
        # ==========================================
        with torch.no_grad():
            model.eval()
            all_feats_val = []
            targets = np.array([])
            mask = np.array([])

            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                images, label, _ = batch[:3]
                images = images.cuda()

                features = model(images)
                all_feats_val.append(features.detach().cpu().numpy())
                targets = np.append(targets, label.cpu().numpy())
                mask = np.append(
                    mask,
                    np.array(
                        [
                            (
                                True
                                if x.item() in range(len(args.train_classes))
                                else False
                            )
                            for x in label
                        ]
                    ),
                )

        # Clustering evaluation
        (
            img_all_acc_test,
            img_old_acc_test,
            img_new_acc_test,
            img_agglo_score,
            estimated_k,
        ) = test_agglo(epoch, all_feats_val, targets, mask, "Test/ACC", args)

        if args.wandb:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "test/all": img_all_acc_test,
                    "test/base": img_old_acc_test,
                    "test/novel": img_new_acc_test,
                    "score/agglo": img_agglo_score,
                    "score/estimated_k": estimated_k,
                    "loss/total": loss_record.avg,
                    "loss/cms_unsup": loss_cms_unsup_record.avg,
                    "loss/cms_sup": loss_cms_sup_record.avg,
                    "loss/arcface": loss_arc_record.avg,
                    "train/lr": current_lr,
                },
                step=epoch,
            )

        print(
            "Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(
                img_all_acc_test, img_old_acc_test, img_new_acc_test
            )
        )

        # Step scheduler
        exp_lr_scheduler.step()
        torch.save(model.state_dict(), args.model_path)

        if img_agglo_score > best_agglo_img_score:
            torch.save(
                {"k": estimated_k, "model_state_dict": model.state_dict()},
                args.model_path[:-3] + f"_best.pt",
            )
            best_agglo_img_score = img_agglo_score
            print(f"New best model saved! Agglo score: {best_agglo_img_score:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CMS + ArcFace Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--eval_funcs", nargs="+", help="Which eval functions to use", default=["v2"]
    )

    parser.add_argument("--warmup_model_dir", type=str, default=None)
    parser.add_argument("--exp_root", type=str, default=CMS_ROOT + "log")
    parser.add_argument("--pretrain_path", type=str, default=CMS_ROOT + "models")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar100",
        help="options: cifar10, cifar100, scars, cub, aircraft, herbarium_19",
    )
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--use_ssb_splits", type=bool, default=True)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eta_min", type=float, default=1e-3)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_views", default=2, type=int)
    parser.add_argument("--contrast_unlabel_only", type=bool, default=False)

    # CMS hyperparameters
    parser.add_argument(
        "--sup_con_weight",
        type=float,
        default=0.35,
        help="Weight for supervised contrastive loss",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", default=8, type=int)

    # ArcFace hyperparameters
    parser.add_argument(
        "--arcface_weight",
        type=float,
        default=0.3,
        help="Weight for ArcFace loss (0.1-0.5 recommended)",
    )
    parser.add_argument(
        "--arcface_s", type=float, default=64.0, help="ArcFace scale parameter"
    )
    parser.add_argument(
        "--arcface_m", type=float, default=0.5, help="ArcFace margin parameter"
    )

    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Flag to log at wandb")

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_class_splits(args)

    args.feat_dim = 768
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=["cms_arcface"])
    print(f"Using evaluation function {args.eval_funcs[0]} to print results")

    if args.wandb:
        os.environ["WANDB_WATCH"] = "all"
        wandb.init(
            project="CMS-ArcFace",
            settings=wandb.Settings(_disable_stats=False),
        )
        wandb.config.update(args)

    print(f"\n{'='*80}")
    print(f"CMS + ArcFace Configuration:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Labeled classes: {args.num_labeled_classes}")
    print(f"  CMS supervised weight (lambda_sup): {args.sup_con_weight}")
    print(f"  ArcFace weight (lambda_arc): {args.arcface_weight}")
    print(f"  ArcFace scale (s): {args.arcface_s}")
    print(f"  ArcFace margin (m): {args.arcface_m}")
    print(f"{'='*80}\n")

    # --------------------
    # MODEL
    # --------------------
    model = DINO_with_ArcFace(args)
    model = nn.DataParallel(model).to(device)

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(
        args.transform, image_size=args.image_size, args=args
    )
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=args.n_views
    )

    # --------------------
    # DATASETS
    # --------------------
    if args.inductive:
        (
            train_dataset,
            test_dataset,
            unlabelled_train_examples_test,
            val_datasets,
            datasets,
        ) = get_datasets_with_gcdval(
            args.dataset_name, train_transform, test_transform, args
        )
    else:
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets = (
            get_datasets(args.dataset_name, train_transform, test_transform, args)
        )

    # --------------------
    # SAMPLER
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [
        1 if i < label_len else label_len / unlabelled_len
        for i in range(len(train_dataset))
    ]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset)
    )

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
    )
    train_loader_memory = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    if args.inductive:
        test_loader_labelled = DataLoader(
            val_datasets,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        test_loader_labelled = DataLoader(
            test_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )

    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, train_loader_memory, args)
