from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from model.model import ClassificationModel

def get_trainer(exp_name):
    earlystopper = EarlyStopping(monitor="val_f1_macro", patience=4, min_delta=0.001)
    progress_bar = TQDMProgressBar(refresh_rate=10)
    logger = CSVLogger("logs", name=exp_name)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=20,
        callbacks=[progress_bar, earlystopper],
        logger=logger,
        enable_checkpointing=False,
    )
    
    return trainer


def train(exp_name, datamodule, out_size, transformer_name, do_proportions=True):
    datamodule.prepare_data()
    datamodule.setup()
    proportions = datamodule.train_data.class_proportion() if do_proportions else None
    
    model = ClassificationModel(out_size=out_size, transformer_name=transformer_name, lr=1e-5, class_proportion=proportions)
    
    trainer = get_trainer(exp_name)
    
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    
    return trainer, model