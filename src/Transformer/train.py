import os
import importlib
from argparse import ArgumentParser

from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pathlib import Path

from models import module

if __name__ == "__main__":
    # ----------------------------------- #
    # ----------------------------------- #
    parser = ArgumentParser()

    # Program specific
    parser.add_argument("--log_dir", type=str, help="Directory to which experiment logs will be written", required=True)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--dataset", type=str, help="Dataset module", required=True)
    parser.add_argument("--gradient_clip_val", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int)

    # Model specific
    parser = module.get_model_def().add_model_specific_args(parser)

    program_args, _ = parser.parse_known_args()

    # Data module specific 
    data_module = importlib.import_module(f'datasets.{program_args.dataset}')

    parser = data_module.get_datamodule_def().add_datamodule_specific_args(parser)

    args = parser.parse_args()

    dict_args = vars(args)

    # Gets our specific model
    model = module.get_model(**dict_args)

    # Gets our specific dataloader
    dm = data_module.get_datamodule(**dict_args)

    logger = TensorBoardLogger(Path(args.log_dir), name=args.model)
    trainer = Trainer(
        logger=logger,
        detect_anomaly=True,
        # fast_dev_run=args.fast_dev_run,
        # track_grad_norm=args.track_grad_norm,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=200,
        # log_gpu_memory=args.log_gpu_memory,
        # overfit_batches=args.overfit_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # val_check_interval=args.val_check_interval,
        # profiler=args.profiler,
        # progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        num_nodes=args.gpus,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=10),
                   LearningRateMonitor(logging_interval="epoch")]
    )


    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, dm)
    

