from dataset.multiemo import MultiEmoDataset
from dataset.persent import PerSenTDataset
from dataset.datamodule import SentimentDataModule
from dataset.multidatamodule import MultiFileDataModule
from model.trainer import train, get_trainer
import torch
from os.path import isdir
from model.model import ClassificationModel
import json
import time
from tqdm import tqdm
import numpy as np

def test_persent_datamodule():
    datadir = "../data/PerSenT"
    train_filepath = "train.csv"
    dev_filepath = "dev.csv"
    test_filepath = "fixed_test.csv"
    datamodule = SentimentDataModule(
        datadir=datadir,
        train_filepath=train_filepath,
        dev_filepath=dev_filepath,
        test_filepath=test_filepath,
        dataset_class=PerSenTDataset,
        batch_size=16,
    )
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader()
    for batch in loader:
        x, y = batch
        print(x)
        print(y)
        break
    
def test_multiemo_datamodule():
    datadir = "../data/multiemo2"
    train_filepath = "all.text.train.en.txt"
    dev_filepath = "all.text.dev.en.txt"
    test_filepath = "all.text.test.en.txt"
    datamodule = SentimentDataModule(
        datadir=datadir,
        train_filepath=train_filepath,
        dev_filepath=dev_filepath,
        test_filepath=test_filepath,
        dataset_class=MultiEmoDataset,
        batch_size=16,
    )
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader()
    for batch in loader:
        x, y = batch
        print(x)
        print(y)
        break

def test_multiemo_dataset():
    # en, pl, ger, spa
    filepath = "all.text.train.en.txt"
    datadir = "../data/multiemo2"
    
    dataset = MultiEmoDataset(datadir, filepath)
    print(dataset[5])

def test_persent_dataset():
    datadir = "../data/PerSenT"
    filepath = "train.csv"
    dataset = PerSenTDataset(datadir, filepath)
    print(dataset[5])
    # print(dataset[15])
    # print(dataset[25])
    
    
def training_persent():
    datadir = "../data/PerSenT"
    train_filepath = "paraphrase_train_different_words.csv"
    dev_filepath = "dev.csv"
    test_filepath = "fixed_test.csv"
    proportions = False
    datamodule = SentimentDataModule(
        datadir=datadir,
        train_filepath=train_filepath,
        dev_filepath=dev_filepath,
        test_filepath=test_filepath,
        dataset_class=PerSenTDataset,
        batch_size=16,
    )
    out_size = 3
    # transformer = "xlm-roberta-base"
    # transformer = "xlm-roberta-large"
    # transformer = "Unbabel/xlm-roberta-comet-small"
    transformer = "sentence-transformers/LaBSE"
    # transformer = "sdadas/polish-distilroberta"
    # transformer = "microsoft/xtremedistil-l6-h256-uncased"
    
    
    exp_name = f"persent-paraphrase_different-{transformer}"
    
    trainer, model = train(exp_name, datamodule, out_size, transformer, proportions)
    
    trainer.test(model=model, datamodule=datamodule)
    
    
def multi_training_persent():
    datadir = "../data/PerSenT"
    train_filepaths = ["train.csv", "new_train_v2.csv"] #"train.csv", "paraphrase_train_normal.csv", "paraphrase_train_different_words.csv", "new_train.csv", "new_train_v2.csv"
    dev_filepath = "dev.csv"
    test_filepath = "fixed_test.csv"
    new_train_filepath = "train+new_v2.csv"
    proportions = False
    datamodule = MultiFileDataModule(
        datadir=datadir,
        train_filepaths=train_filepaths,
        dev_filepath=dev_filepath,
        test_filepath=test_filepath,
        new_train_filepath=new_train_filepath,
        dataset_class=PerSenTDataset,
        batch_size=16,
    )
    out_size = 3
    # transformer = "xlm-roberta-base"
    # transformer = "xlm-roberta-large"
    # transformer = "Unbabel/xlm-roberta-comet-small"
    transformer = "sentence-transformers/LaBSE"
    # transformer = "sdadas/polish-distilroberta"
    # transformer = "microsoft/xtremedistil-l6-h256-uncased"
    
    
    exp_name = f"persent-multi-original+new_v2-{transformer}"
    
    trainer, model = train(exp_name, datamodule, out_size, transformer, proportions)
    
    trainer.test(model=model, datamodule=datamodule)
    
    
def training_multiemo():
    datadir = "../data/multiemo2"
    train_filepath = "all.text.train.en.txt"
    dev_filepath =  train_filepath.replace("train", "dev")
    test_filepath =  train_filepath.replace("train", "test")
    # train_filepath = train_filepath.replace("train", "short_train").replace("txt", "csv")
    proportions = False
    
    datamodule = SentimentDataModule(
        datadir=datadir,
        train_filepath=train_filepath,
        dev_filepath=dev_filepath,
        test_filepath=test_filepath,
        dataset_class=MultiEmoDataset,
        batch_size=16,
    )
    out_size = 4
    # transformer = "xlm-roberta-base"
    # transformer = "xlm-roberta-large"
    # transformer = "Unbabel/xlm-roberta-comet-small"
    transformer = "sentence-transformers/LaBSE"
    # transformer = "sdadas/polish-distilroberta"
    # transformer = "microsoft/xtremedistil-l6-h256-uncased"
    
    
    exp_name = f"testowe-multiemo-{transformer}"
    
    trainer, model = train(exp_name, datamodule, out_size, transformer, proportions)

    trainer.test(model=model, datamodule=datamodule)
    
def multitraining_pipeline(lan="en", repeating_no=5):
    persent = dict(
        datadir = "../data/PerSenT",
        dev_filepath = "dev.csv",
        test_filepath = "fixed_test.csv",
        out_size = 3,
        trainpaths_dict = {
            "original": "train.csv", 
            "normal": "paraphrase_train_normal.csv",
            "different": "paraphrase_train_different_words.csv",
            "new": "new_train.csv",
            "new_v2": "new_train_v2.csv"
        },
        exp_name = "persent-{comb}-{transformer}",
        dataset_class=PerSenTDataset,
    )
    original_trainpath = f"all.text.train.{lan}.txt" 
    multiemo = dict(
        datadir = "../data/multiemo2",
        dev_filepath =  f"all.text.dev.{lan}.txt",
        test_filepath =  f"all.text.test.{lan}.txt",
        out_size = 4,
        trainpaths_dict = {
            "original": original_trainpath, 
            "normal": original_trainpath.replace("train", "paraphrase_train").replace(".txt", "_normal.csv"),
            "different": original_trainpath.replace("train", "paraphrase_train").replace(".txt", "_different_words.csv"),
            "new": original_trainpath.replace("train", "new_train").replace(".txt", ".csv"),
            "new_v2": original_trainpath.replace("train", "new_train_v2").replace(".txt", ".csv"),
        },
        exp_name = "multiemo-{comb}-{transformer}",
        dataset_class=MultiEmoDataset,
    )
    proportions = True
    combinations = [
            "original",
            "original+normal",
            "original+different",
            "original+normal+different",
            "original+new",
            "original+new_v2",
            "original+new+new_v2",
            "original+normal+different+new+new_v2",
            
            "normal",
            "different",
            "new",
            "new_v2",
        ]
    transformers = [
        ("Unbabel/xlm-roberta-comet-small", 16),
        ("xlm-roberta-base", 16),
        ("sentence-transformers/LaBSE", 16),
        ("microsoft/xtremedistil-l6-h256-uncased", 16)
        # ("xlm-roberta-large", 4), # jeśli dajemy to dodać akumulacje gradientu
    ]
        # ("sdadas/polish-distilroberta", 16),
    
    for transformer, batch_size in transformers:
        for comb in combinations:
            for dataset in (persent, multiemo):
                splits = comb.split("+")
                new_train_filepath = comb + ".csv"
                exp_name = dataset["exp_name"].format(comb=comb, transformer=transformer)
                print(exp_name)
                if isdir("logs/" + exp_name):
                    continue
                train_filepaths = [
                    dataset["trainpaths_dict"][name] for name in splits
                ]
                datamodule = MultiFileDataModule(
                    datadir=dataset["datadir"],
                    train_filepaths=train_filepaths,
                    dev_filepath=dataset["dev_filepath"],
                    test_filepath=dataset["test_filepath"],
                    new_train_filepath=new_train_filepath,
                    dataset_class=dataset["dataset_class"],
                    batch_size=batch_size,
                )
                
                for iteration in range(repeating_no):
                    torch.manual_seed(iteration*42)
                    
                    trainer, model = train(exp_name, datamodule, dataset["out_size"], transformer, proportions)
                    trainer.test(model=model, datamodule=datamodule)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def time_test(lan="en", iteration=100):
    persent = dict(
        datadir = "../data/PerSenT",
        train_filepath =  "train.csv",
        dev_filepath = "dev.csv",
        test_filepath = "fixed_test.csv",
        out_size = 3,
        exp_name = "persent-time-{transformer}",
        name = "persent",
        dataset_class=PerSenTDataset,
    )
    multiemo = dict(
        datadir = "../data/multiemo2",
        dev_filepath =  f"all.text.dev.{lan}.txt",
        test_filepath =  f"all.text.test.{lan}.txt",
        out_size = 4,
        train_filepath =  f"all.text.train.{lan}.txt" ,
        exp_name = "multiemo-time-{transformer}",
        name = "multiemo",
        dataset_class=MultiEmoDataset,
    )
    transformers = [
        "Unbabel/xlm-roberta-comet-small",
        "xlm-roberta-base",
        # "sentence-transformers/LaBSE",
        "microsoft/xtremedistil-l6-h256-uncased",
        # "xlm-roberta-large",
    ]
    data_to_save = {"multiemo": {}, "persent": {}}
    json_file = "../data/times_proper_batch_inside.json"
    
    batch_size = 16
    
    for dataset in (persent, multiemo):
        datamodule = SentimentDataModule(
            datadir=dataset["datadir"],
            train_filepath=dataset["train_filepath"],
            dev_filepath=dataset["dev_filepath"],
            test_filepath=dataset["test_filepath"],
            dataset_class=dataset["dataset_class"],
            batch_size=batch_size,
        )
        datamodule.prepare_data()
        datamodule.setup()
                
        for transformer in transformers:
            exp_name = dataset["exp_name"].format(transformer=transformer)
            model = ClassificationModel(out_size=dataset["out_size"], transformer_name=transformer,
                                        lr=1e-5, class_proportion=True)
            model = model.to("cuda")
            # print(get_n_params(model))
            # trainer = get_trainer(exp_name)
            # print(model)
            # trainer.fit(model)
        # break
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for X, Y in datamodule.test_dataloader():
                break
            X = list(X)
            timings=np.zeros((iteration,1))
            #GPU-WARM-UP
            for _ in range(10):
                _ = model(X)
            # MEASURE PERFORMANCE
            with torch.no_grad():
                for rep in tqdm(range(iteration)):
                    starter.record()
                    _ = model(X)
                    ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time

            mean_syn = np.sum(timings) / (iteration*batch_size)
            std_syn = np.std(timings/batch_size)
            
            data_to_save[dataset["name"]][transformer] = {
                "time_mean": mean_syn,
                "time_std": std_syn,
            }
            
            with open(json_file, "w") as f:
                json.dump(data_to_save, f)
                

if __name__ == "__main__":
    # test_multiemo_dataset()
    # test_persent_dataset()
    # test_persent_datamodule()
    # test_multiemo_datamodule()
    # training_persent()
    # training_multiemo()
    # multi_training_persent()
    # multitraining_pipeline()
    time_test(iteration=2000)
    
    
    