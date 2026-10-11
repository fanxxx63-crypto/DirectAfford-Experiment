"""Reproducible LASO runner for this repository's unchanged DirectAffordModel.

The method is fixed: frozen Hugging Face CLIP pooler, 10 * cosine logits,
soft-label BCE, and AdamW over geometry AND regression-head parameters.
No geometric pretrained weights, scheduler, AMP, or early stopping are used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import random
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import DirectAffordModel

COUNTS = {"train": 16120, "val": 1215, "test": 2416}
ROOT = Path(__file__).resolve().parents[1]
FORMAT = "directafford-experiment.laso.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def atomic_bytes(path, content):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode())


def save_checkpoint(path, payload):
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def copy_checkpoint(source, destination):
    temporary = destination.with_suffix(".pending")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        require(torch.cuda.is_available(), "CUDA checkpoint requires CUDA to resume")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def worker_seed(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class LasoDataset(Dataset):
    """Strict official files, with no synthetic or missing-data fallback."""

    def __init__(self, root, split, query_mode, normalize=False, limit=None):
        self.split, self.query_mode, self.normalize = split, query_mode, normalize
        self.hashes = {name: digest(root / name) for name in
                       (f"anno_{split}.pkl", f"objects_{split}.pkl", "Affordance-Question.csv")}
        with (root / f"anno_{split}.pkl").open("rb") as stream:
            self.annotations = pickle.load(stream)
        with (root / f"objects_{split}.pkl").open("rb") as stream:
            self.objects = pickle.load(stream)
        require(isinstance(self.annotations, list), "Official annotations must be a list")
        require(isinstance(self.objects, dict), "Official objects must be a dictionary")
        require(len(self.annotations) == COUNTS[split], f"Wrong {split} sample count")
        self.questions = {}
        with (root / "Affordance-Question.csv").open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            require(all(k in fields for k in ("Object", "Affordance", "Question0")),
                    "Question CSV lacks required columns")
            columns = sorted((k for k in fields if k.startswith("Question")
                              and k[8:].isdigit()), key=lambda k: int(k[8:]))
            for row in reader:
                key = (row["Object"].strip().lower(), row["Affordance"].strip())
                questions = [row[k].strip() for k in columns
                             if row[k] and row[k].strip() and row[k].strip().lower() != "nan"]
                require(key not in self.questions and row["Question0"].strip()
                        and row["Question0"].strip().lower() != "nan" and questions,
                        f"Duplicate/missing canonical question: {key}")
                self.questions[key] = questions
        seen = set()
        for item in self.annotations:
            require(isinstance(item, dict) and all(k in item for k in
                    ("shape_id", "class", "affordance", "mask")), "Invalid annotation schema")
            sid = str(item["shape_id"])
            require(isinstance(item["affordance"], str), "Affordance must be a string")
            key = (str(item["class"]).lower(), item["affordance"])
            sample_id = sid + "__" + item["affordance"]
            require(sample_id not in seen and sid in self.objects, f"Missing/duplicate shape: {sample_id}")
            seen.add(sample_id)
            require(key in self.questions, f"Question not found: {key}")
            points = np.asarray(self.objects[sid])
            mask = np.asarray(item["mask"])
            require(points.shape == (2048, 3) and mask.shape in ((2048,), (2048, 1)),
                    f"Expected 2048 xyz points and aligned mask: {sample_id}: {points.shape}/{mask.shape}")
            require(np.isfinite(points).all() and np.isfinite(mask).all(), f"Nonfinite input: {sample_id}")
            require(((mask >= 0) & (mask <= 1)).all(), f"Mask outside [0,1]: {sample_id}")
            if split != "train":
                binary = mask >= 0.5
                require(binary.any() and not binary.all(), f"Undefined sample AUC: {sample_id}")
        require(limit is None or 0 < limit <= len(self.annotations), "Invalid debug subset size")
        self.count = len(self.annotations) if limit is None else limit
        self.debug = limit is not None

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        item = self.annotations[index]
        points = np.array(self.objects[str(item["shape_id"])], dtype=np.float32, copy=True)
        mask = np.asarray(item["mask"], dtype=np.float32).reshape(2048)
        if self.normalize:
            points -= points.mean(axis=0, keepdims=True)
            radius = np.linalg.norm(points, axis=1).max()
            require(radius > 1e-6, "Degenerate point cloud cannot be normalized")
            points /= radius
        questions = self.questions[(str(item["class"]).lower(), item["affordance"])]
        query = item["affordance"] if self.query_mode == "affordance" else questions[0]
        if self.split == "train" and self.query_mode == "mixed" and len(questions) > 1:
            query = questions[0] if random.random() < 0.5 else random.choice(questions[1:])
        return {"points": torch.from_numpy(points), "mask": torch.from_numpy(mask),
                "query": query, "sample_id": str(item["shape_id"]) + "__" + item["affordance"]}


def loader(dataset, args, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      drop_last=shuffle, num_workers=args.workers, worker_init_fn=worker_seed,
                      generator=generator, persistent_workers=False,
                      pin_memory=args.device == "cuda")


def scores_and_logits(model, batch, device):
    points = batch["points"].to(device)
    text = model.get_text_embeddings(list(batch["query"]), device)
    features = model(points)
    scores = (features * text.unsqueeze(1)).sum(dim=-1)
    require(torch.isfinite(scores).all().item(), "Nonfinite model output")
    return scores, 10.0 * scores


def measure(model, dataset, args, save_scores=False):
    """Evaluation gets its own RNG stream and never advances the training RNG."""
    state, was_training = rng_state(), model.training
    try:
        set_seed(args.seed)
        model.eval()
        records, outputs = [], []
        with torch.inference_mode():
            for batch in loader(dataset, args, generator=torch.Generator().manual_seed(args.seed)):
                scores, logits = scores_and_logits(model, batch, args.device)
                probabilities = torch.sigmoid(logits.double()).cpu().numpy()
                raw = scores.double().cpu().numpy()
                for i, sid in enumerate(batch["sample_id"]):
                    mask = batch["mask"][i].numpy().astype(np.float64)
                    binary = mask >= 0.5
                    require(binary.any() and not binary.all(), f"Undefined AUC: {sid}")
                    records.append({"sample_id": sid, "query": batch["query"][i],
                                    "num_points": 2048, "n_pos": int(binary.sum()),
                                    "auc_raw": float(roc_auc_score(binary, raw[i])),
                                    "mae_prob": float(np.abs(probabilities[i] - mask).mean()),
                                    "auc_skipped": False})
                    if save_scores:
                        outputs.append({"sample_id": sid, "points": batch["points"][i].numpy(),
                                        "mask": mask.astype(np.float32), "scores": raw[i],
                                        "logits": logits[i].double().cpu().numpy(),
                                        "probabilities": probabilities[i]})
        require(len(records) == len(dataset), "Evaluation lost samples")
        metrics = {k: float(np.mean([r[k] for r in records])) for k in ("auc_raw", "mae_prob")}
        metrics.update(n_samples=len(records), n_auc_scored=len(records), n_auc_skipped=0)
        return metrics, records, outputs
    finally:
        restore_rng(state)
        model.train(was_training)
        model.text_encoder.eval()


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"commit": commit, "model_sha256": digest(ROOT / "src/model.py"),
            "runner_sha256": digest(__file__), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("transformers", "scikit-learn", "tokenizers")},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_model(args):
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    model = DirectAffordModel(text_encoder_name=args.text_model).to(args.device)
    model.text_encoder.eval()
    return model


def text_assets(args):
    """A local snapshot pins tokenizer behavior as well as the frozen weights."""
    directory = Path(args.text_model)
    require(directory.is_dir(), "Use --text-model with a local CLIP snapshot for reproducible runs")
    files = sorted(p for p in directory.iterdir() if p.is_file()
                   and p.suffix in (".json", ".txt", ".bin", ".safetensors"))
    require(files and (directory / "config.json").is_file(), "Incomplete local CLIP snapshot")
    return {p.name: digest(p) for p in files}


def configuration(args):
    return {k: getattr(args, k) for k in ("data_root", "text_model", "query_mode", "normalize",
            "seed", "epochs", "batch_size", "workers", "device", "limit_train", "limit_val")}


def publish_training_record(directory, payload):
    history = payload["history"]
    atomic_bytes(directory / "metrics.jsonl", ("\n".join(canonical_json(x) for x in history) + "\n").encode())
    write_json(directory / "summary.json", {"epochs_run": payload["epoch"],
               "planned_epochs": payload["config"]["epochs"],
               "complete": payload["epoch"] == payload["config"]["epochs"],
               "best_epoch": payload["best_epoch"], "best_val_auc": payload["best_auc"],
               "checkpoint_sha256": digest(directory / "best_auc.pth"), "debug": payload["debug"],
               "test_split_accessed": False, "updated_utc": now()})


def train(args):
    cfg = configuration(args)
    config_hash = hashlib.sha256(canonical_json(cfg).encode()).hexdigest()
    directory = Path(args.output_dir)
    if args.resume:
        require(directory.is_dir() and Path(args.resume).resolve() == (directory / "last.pth").resolve(),
                "Resume must use this output directory's last.pth")
    else:
        require(not directory.exists(), "Output already exists; choose a new directory or explicitly resume")
    datasets = {split: LasoDataset(Path(args.data_root), split, args.query_mode, args.normalize,
                                  getattr(args, "limit_" + split)) for split in ("train", "val")}
    require(len(datasets["train"]) >= args.batch_size, "Train subset is smaller than one full batch")
    clip_hashes = text_assets(args)
    model = load_model(args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    info = provenance()
    info["text_encoder_config"] = model.text_encoder.config.to_dict()
    data_hashes = {split: ds.hashes for split, ds in datasets.items()}
    history, best, best_epoch, start = [], -1.0, 0, 1
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        require(payload["format"] == FORMAT and payload["config"] == cfg, "Resume configuration mismatch")
        require(payload["data_sha256"] == data_hashes, "Resume dataset changed")
        require(payload["text_assets_sha256"] == clip_hashes, "Resume CLIP assets changed")
        require(all(payload["source"][k] == info[k] for k in ("model_sha256", "runner_sha256")),
                "Resume implementation changed")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        # load_state_dict casts Adam moments to their parameter devices and
        # retains CPU step counters for non-capturable AdamW.
        history, best, best_epoch = payload["history"], payload["best_auc"], payload["best_epoch"]
        start = payload["epoch"] + 1
        generator.set_state(payload["loader_rng"].cpu())
        restore_rng(payload["rng"])
        if best_epoch == payload["epoch"]:
            copy_checkpoint(directory / "last.pth", directory / "best_auc.pth")
        else:
            selected = torch.load(directory / "best_auc.pth", map_location="cpu")
            require(selected["epoch"] == best_epoch and selected["best_auc"] == best,
                    "Best checkpoint does not match resume state")
        publish_training_record(directory, payload)
        if start > args.epochs:
            print("Completed checkpoint recovered; training already finished.", flush=True)
            return
    else:
        directory.mkdir(parents=True)
        write_json(directory / "run_meta.json", {"config": cfg, "source": info, "data_sha256": data_hashes,
                   "config_sha256": config_hash,
                   "text_assets_sha256": clip_hashes,
                   "started_utc": now(), "debug": any(ds.debug for ds in datasets.values()),
                   "test_split_accessed": False, "train_samples": len(datasets["train"]),
                   "val_samples": len(datasets["val"]), "method": "original_model_cosine10_bce",
                   "optimizer": {"type": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
                   "scheduler": None, "precision": "float32", "early_stopping": False})
    for epoch in range(start, args.epochs + 1):
        model.train()
        model.text_encoder.eval()
        started, total, batches = time.monotonic(), 0.0, 0
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in loader(datasets["train"], args, shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            _, logits = scores_and_logits(model, batch, args.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["mask"].to(args.device))
            require(torch.isfinite(loss).item(), "Nonfinite training loss")
            loss.backward()
            finite_gradients = [torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.requires_grad and p.grad is not None]
            require(torch.stack(finite_gradients).all().item(), "Nonfinite training gradient")
            optimizer.step()
            total += loss.item()
            batches += 1
            if batches % args.log_every == 0:
                print(f"epoch={epoch}/{args.epochs} batch={batches} loss={loss.item():.6f}", flush=True)
        val, _, _ = measure(model, datasets["val"], args)
        improved = val["auc_raw"] > best
        if improved:
            best, best_epoch = val["auc_raw"], epoch
        row = {"epoch": epoch, "train_loss": total / batches, "optimizer_steps": batches,
               "validation": val, "best_epoch": best_epoch, "seconds": time.monotonic() - started,
               "peak_cuda_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if args.device == "cuda" else None,
               "completed_utc": now()}
        history.append(row)
        payload = {"format": FORMAT, "epoch": epoch, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": None, "rng": rng_state(),
                   "loader_rng": generator.get_state(), "config": cfg, "source": info,
                   "config_sha256": config_hash,
                   "text_assets_sha256": clip_hashes,
                   "data_sha256": data_hashes, "history": history, "best_auc": best,
                   "best_epoch": best_epoch, "debug": any(ds.debug for ds in datasets.values())}
        # Commit last first: a resume can recover best from it if interrupted here.
        save_checkpoint(directory / "last.pth", payload)
        if improved:
            copy_checkpoint(directory / "last.pth", directory / "best_auc.pth")
        publish_training_record(directory, payload)
        print(canonical_json(row), flush=True)
        if args.stop_after_epoch == epoch:
            print("Debug run paused after a complete checkpoint; resume with the same planned configuration.", flush=True)
            return


def evaluate(args):
    require(args.checkpoint is not None, "Evaluation requires --checkpoint")
    require(args.split != "test" or args.allow_test, "Test requires explicit --allow-test")
    directory = Path(args.output_dir)
    require(not directory.exists(), "Refusing to overwrite evaluation output")
    payload = torch.load(args.checkpoint, map_location="cpu")
    require(payload["format"] == FORMAT, "Wrong checkpoint format")
    saved = payload["config"]
    for key in ("query_mode", "normalize", "seed", "text_model", "batch_size"):
        require(saved[key] == getattr(args, key), f"Evaluation {key} differs from training")
    require(payload["source"]["model_sha256"] == digest(ROOT / "src/model.py"), "Model source changed")
    require(payload["text_assets_sha256"] == text_assets(args), "Evaluation CLIP assets changed")
    if args.split == "test":
        require(args.limit_train is None and args.limit_val is None, "Test cannot use debug subset flags")
        require(not payload["debug"], "Debug-trained checkpoints cannot produce formal test results")
        require(payload["epoch"] == payload["best_epoch"], "Test requires validation-selected checkpoint")
        summary = json.loads((Path(args.checkpoint).parent / "summary.json").read_text())
        require(summary["complete"] and summary["checkpoint_sha256"] == digest(args.checkpoint),
                "Formal test requires completed training and the recorded best checkpoint")
    dataset = LasoDataset(Path(args.data_root), args.split, args.query_mode, args.normalize,
                          args.limit_val if args.split == "val" else None)
    for name, value in dataset.hashes.items():
        if name in payload["data_sha256"]["val"]:
            require(payload["data_sha256"]["val"][name] == value, f"Dataset changed: {name}")
    model = load_model(args)
    model.load_state_dict(payload["model"], strict=True)
    started = time.monotonic()
    metrics, rows, scores = measure(model, dataset, args, args.save_scores)
    require(args.split != "test" or metrics["n_samples"] == metrics["n_auc_scored"] == 2416,
            "Incomplete official test")
    directory.mkdir(parents=True)
    with (directory / "metrics_per_sample.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "predictions.json", {"metrics": metrics, "samples": rows})
    if args.save_scores:
        np.savez_compressed(directory / "scores.npz", sample_ids=np.array([s["sample_id"] for s in scores]),
                            **{k: np.stack([s[k] for s in scores]) for k in
                               ("points", "mask", "scores", "logits", "probabilities")})
    write_json(directory / "meta.json", {"checkpoint_sha256": digest(args.checkpoint),
               "checkpoint_epoch": payload["epoch"], "checkpoint_compatibility": "strict",
               "training_source": payload["source"], "evaluation_source": provenance(),
               "text_assets_sha256": payload["text_assets_sha256"],
               "config": saved, "split": args.split, "query_mode": "canonical" if args.query_mode == "mixed" else args.query_mode,
               "config_sha256": payload["config_sha256"],
               "inference_batch_size": args.batch_size, "inference_workers": args.workers,
               "inference_device": args.device, "evaluation_seed": args.seed,
               "normalize": args.normalize, "num_points": 2048, "n_samples": len(dataset),
               "n_auc_skipped": 0, "debug": dataset.debug or payload["debug"],
               "full_test_split": args.split == "test" and not dataset.debug,
               "calibration": "none", "data_sha256": dataset.hashes,
               "auc_definition": "mean per-sample ROC AUC(raw cosine, soft mask >= 0.5)",
               "mae_definition": "mean per-sample MAE(sigmoid(10 * cosine), soft mask)",
               "seconds": time.monotonic() - started, "completed_utc": now()})
    print(canonical_json(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--data-root", required=True, help="Directory directly containing anno_train.pkl etc.")
    parser.add_argument("--output-dir")
    parser.add_argument("--text-model", required=True, help="Local openai/clip-vit-base-patch32 snapshot")
    parser.add_argument("--query-mode", choices=("canonical", "mixed", "affordance"), default="affordance")
    parser.add_argument("--normalize", action="store_true", help="Explicit protocol change; default preserves raw coordinates")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--resume", help="Explicit epoch-boundary resume from this run's last.pth")
    parser.add_argument("--stop-after-epoch", type=int, help="Debug subsets only: stop cleanly without changing planned epochs")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    require(min(args.epochs, args.batch_size, args.log_every) > 0 and args.workers >= 0, "Invalid run dimensions")
    require(args.phase == "preflight" or args.output_dir, "Training/evaluation requires --output-dir")
    if args.stop_after_epoch is not None:
        require(args.phase == "train" and (args.limit_train is not None or args.limit_val is not None)
                and 1 <= args.stop_after_epoch <= args.epochs,
                "--stop-after-epoch requires a debug training subset and a valid epoch")
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    if Path(args.text_model).exists():
        args.text_model = str(Path(args.text_model).expanduser().resolve())
    set_seed(args.seed)
    if args.phase == "train":
        train(args)
    elif args.phase == "evaluate":
        evaluate(args)
    else:
        report = {"source": provenance(), "config": configuration(args), "test_split_accessed": False,
                  "text_assets_sha256": text_assets(args)}
        for split in ("train", "val"):
            dataset = LasoDataset(Path(args.data_root), split, args.query_mode, args.normalize,
                                  getattr(args, "limit_" + split))
            report[split] = {"samples": len(dataset), "debug": dataset.debug, "sha256": dataset.hashes}
        model = load_model(args)
        require(all(not p.requires_grad for p in model.text_encoder.parameters()), "CLIP must be frozen")
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
