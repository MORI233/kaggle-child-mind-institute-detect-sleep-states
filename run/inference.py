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
    # model3 = get_model(
    # cfg,
    # feature_dim=len(cfg.features),
    # n_classes=len(cfg.labels),
    # num_timesteps=num_timesteps // cfg.downsample_rate,
    # test=True,
    # )
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

    model6 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model7 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model8 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model9 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model10 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    
    model11 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )

    model12 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )



    model_1 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model_2 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model_3 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model_4 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model_5 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )

    model_6 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model_7 = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
        test=True,
    )
    model_8 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model_9 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )
    model_10 = get_model(
    cfg,
    feature_dim=len(cfg.features),
    n_classes=len(cfg.labels),
    num_timesteps=num_timesteps // cfg.downsample_rate,
    test=True,
    )    


    # load weights
    if cfg.weight is not None:
        weight_path = (
            Path(cfg.dir.model_dir) / cfg.weight.exp_name / cfg.weight.run_name / "best_model.pth"
        )
        weight_path1 = "/kaggle/input/d/daikaizhai/cmi-model/exp011/single/best_model.pth"
        model1.load_state_dict(torch.load(weight_path1))
        print('load weight from "{}"'.format(weight_path1))
        weight_path2 = "/kaggle/input/d/daikaizhai/cmi-model/exp014/single/best_model.pth"
        model2.load_state_dict(torch.load(weight_path2))
        print('load weight from "{}"'.format(weight_path2))
        # weight_path3 = "/kaggle/input/d/daikaizhai/cmi-model/exp015/single/best_model.pth"
        # model3.load_state_dict(torch.load(weight_path3))
        # print('load weight from "{}"'.format(weight_path3))
        weight_path4 = "/kaggle/input/d/daikaizhai/cmi-model/exp018/single/best_model.pth"
        model4.load_state_dict(torch.load(weight_path4))
        print('load weight from "{}"'.format(weight_path4))
        weight_path5 = "/kaggle/input/d/daikaizhai/cmi-model/exp019/single/best_model.pth"
        model5.load_state_dict(torch.load(weight_path5))
        print('load weight from "{}"'.format(weight_path5))
        weight_path6 = "/kaggle/input/d/daikaizhai/cmi-model/exp020/single/best_model.pth"
        model6.load_state_dict(torch.load(weight_path6))
        print('load weight from "{}"'.format(weight_path6))
        weight_path7 = "/kaggle/input/d/daikaizhai/cmi-model/exp021/single/best_model.pth"
        model7.load_state_dict(torch.load(weight_path7))
        print('load weight from "{}"'.format(weight_path7))
        weight_path8 = "/kaggle/input/d/daikaizhai/cmi-model/exp023/single/best_model.pth"
        model8.load_state_dict(torch.load(weight_path8))
        print('load weight from "{}"'.format(weight_path8))
        weight_path9 = "/kaggle/input/d/daikaizhai/cmi-model/exp025/single/best_model.pth"
        model9.load_state_dict(torch.load(weight_path9))
        print('load weight from "{}"'.format(weight_path9))
        weight_path10 = "/kaggle/input/d/daikaizhai/cmi-model/exp027/single/best_model.pth"
        model10.load_state_dict(torch.load(weight_path10))
        print('load weight from "{}"'.format(weight_path10))
        weight_path11 = "/kaggle/input/d/daikaizhai/cmi-model/exp105/single/best_model.pth"
        model11.load_state_dict(torch.load(weight_path11))
        print('load weight from "{}"'.format(weight_path11))
        weight_path12 = "/kaggle/input/d/daikaizhai/cmi-model/exp106/single/best_model.pth"
        model12.load_state_dict(torch.load(weight_path12))
        print('load weight from "{}"'.format(weight_path12))

        weight_path_1 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold0.pth"
        model_1.load_state_dict(torch.load(weight_path_1))
        print('load weight from "{}"'.format(weight_path1))
        weight_path_2 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold1.pth"
        model_2.load_state_dict(torch.load(weight_path_2))
        print('load weight from "{}"'.format(weight_path_2))
        weight_path_3 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold2.pth"
        model_3.load_state_dict(torch.load(weight_path_3))
        print('load weight from "{}"'.format(weight_path_3))
        weight_path_4 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold3.pth"
        model_4.load_state_dict(torch.load(weight_path_4))
        print('load weight from "{}"'.format(weight_path_4))
        weight_path_5 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold4.pth"
        model_5.load_state_dict(torch.load(weight_path_5))
        print('load weight from "{}"'.format(weight_path_5))
        weight_path_6 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold5.pth"
        model_6.load_state_dict(torch.load(weight_path_6))
        print('load weight from "{}"'.format(weight_path_6))
        weight_path_7 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold6.pth"
        model_7.load_state_dict(torch.load(weight_path_7))
        print('load weight from "{}"'.format(weight_path_7))
        weight_path_8 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold7.pth"
        model_8.load_state_dict(torch.load(weight_path_8))
        print('load weight from "{}"'.format(weight_path_8))
        weight_path_9 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold8.pth"
        model_9.load_state_dict(torch.load(weight_path_9))
        print('load weight from "{}"'.format(weight_path_9))
        weight_path_10 = "/kaggle/input/model-sub-exp8-ensemble-1/best_model_fold9.pth"
        model_10.load_state_dict(torch.load(weight_path_10))
        print('load weight from "{}"'.format(weight_path_10))



    
    return model1, model2,model4,model5, model6, model7,model8, model9,model10,model11, model12, model_1, model_2, model_3, model_4,model_5, model_6, model_7,model_8, model_9, model_10


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
        model1, model2,model4,model5, model6, model7,model8, model9,model10,model11, model12, model_1, model_2, model_3, model_4,model_5, model_6, model_7,model_8, model_9, model_10 = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("inference"):
        keys, preds1 = inference(cfg.duration, test_dataloader, model1, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds2 = inference(cfg.duration, test_dataloader, model2, device, use_amp=cfg.use_amp)
    # with trace("inference"):
    #     keys, preds3 = inference(cfg.duration, test_dataloader, model3, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds4 = inference(cfg.duration, test_dataloader, model4, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds5 = inference(cfg.duration, test_dataloader, model5, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds6 = inference(cfg.duration, test_dataloader, model6, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds7 = inference(cfg.duration, test_dataloader, model7, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds8 = inference(cfg.duration, test_dataloader, model8, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds9 = inference(cfg.duration, test_dataloader, model9, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds10 = inference(cfg.duration, test_dataloader, model10, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds11 = inference(cfg.duration, test_dataloader, model11, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds12 = inference(cfg.duration, test_dataloader, model12, device, use_amp=cfg.use_amp)


    with trace("inference"):
        keys, preds_1 = inference(cfg.duration, test_dataloader, model_1, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_2 = inference(cfg.duration, test_dataloader, model_2, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_3 = inference(cfg.duration, test_dataloader, model_3, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_4 = inference(cfg.duration, test_dataloader, model_4, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_5 = inference(cfg.duration, test_dataloader, model_5, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_6 = inference(cfg.duration, test_dataloader, model_6, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_7 = inference(cfg.duration, test_dataloader, model_7, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_8 = inference(cfg.duration, test_dataloader, model_8, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_9 = inference(cfg.duration, test_dataloader, model_9, device, use_amp=cfg.use_amp)
    with trace("inference"):
        keys, preds_10 = inference(cfg.duration, test_dataloader, model_10, device, use_amp=cfg.use_amp)
    
    preds = ((preds1 + preds2 + preds4+ preds5+preds6 + preds7 + preds8+ preds9+ preds10+ preds11+ preds12)/11)*0.3 + ((preds_1 + preds_2 + preds_3 + preds_4+ preds_5+preds_6 + preds_7 + preds_8+ preds_9 + preds_10)/10)*0.7
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
