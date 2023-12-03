from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.conf import InferenceConfig
from src.datamodule import load_chunk_features
from src.dataset.common import get_test_ds
from src.models.base import BaseModel
from src.models.common import get_model
from src.utils.common import nearest_valid_size, trace
from src.utils.post_process import post_process_for_seg


def load_model(cfg: InferenceConfig) -> BaseModel:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model1 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model2 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model3 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model4 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model5 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )

    # load weights
    if cfg.weight is not None:
        # weight_path = (
        #     Path(cfg.dir.model_dir) / cfg.weight.exp_name / cfg.weight.run_name / "best_model.pth"
        # )
        weight_path1 = "/kaggle/input/d/daikaizhai/cmi-model/exp023/single/best_model.pth"
        model1.load_state_dict(torch.load(weight_path1))
        print('load weight from "{}"'.format(weight_path1))
        weight_path2 = "/kaggle/input/d/daikaizhai/cmi-model/exp011/single/best_model.pth"
        model2.load_state_dict(torch.load(weight_path2))
        print('load weight from "{}"'.format(weight_path2))
        weight_path3 = "/kaggle/input/d/daikaizhai/cmi-model/exp018/single/best_model.pth"
        model3.load_state_dict(torch.load(weight_path3))
        print('load weight from "{}"'.format(weight_path3))
        weight_path4 = "/kaggle/input/d/daikaizhai/cmi-model/exp021/single/best_model.pth"
        model4.load_state_dict(torch.load(weight_path4))
        print('load weight from "{}"'.format(weight_path4))
        weight_path5 = "/kaggle/input/d/daikaizhai/cmi-model/exp019/single/best_model.pth"
        model5.load_state_dict(torch.load(weight_path5))
        print('load weight from "{}"'.format(weight_path5))
    return model1, model2, model3,model4,model5


def get_test_dataloader(cfg: InferenceConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = Path(cfg.dir.processed_dir) / cfg.phase
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        duration=cfg.duration,
        feature_names=cfg.features,
        series_ids=series_ids,
        processed_dir=Path(cfg.dir.processed_dir),
        phase=cfg.phase,
    )
    test_dataset = get_test_ds(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
    duration: int, loader: DataLoader, model: BaseModel, device: torch.device, use_amp
) -> tuple[list[str], np.ndarray]:
    model = model.to(device)
    model.eval()

    preds = []
    keys = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                output = model.predict(
                    x,
                    org_duration=duration,
                )
            if output.preds is None:
                raise ValueError("output.preds is None")
            else:
                key = batch["key"]
                preds.append(output.preds.detach().cpu().numpy())
                keys.extend(key)

    preds = np.concatenate(preds)

    return keys, preds  # type: ignore


def make_submission(
    keys: list[str], preds: np.ndarray, score_th, distance
) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds,  # type: ignore
        score_th=score_th,
        distance=distance,  # type: ignore
    )

    return sub_df


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("load model"):
        model1,model2,model3,model4,model5 = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("inference"):
        keys, preds1 = inference(cfg.duration, test_dataloader, model1, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds2 = inference(cfg.duration, test_dataloader, model2, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds3 = inference(cfg.duration, test_dataloader, model3, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds4 = inference(cfg.duration, test_dataloader, model4, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds5 = inference(cfg.duration, test_dataloader, model5, device, use_amp=cfg.use_amp)
    preds = (preds1 + preds2 + preds3+ preds4+ preds5) / 5
    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            score_th=cfg.pp.score_th,
            distance=cfg.pp.distance,
        )
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
