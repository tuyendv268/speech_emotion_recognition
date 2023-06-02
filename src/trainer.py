from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from yaml.loader import SafeLoader
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from tqdm import tqdm
from time import time
from src import utils
import numpy as np
from torch import nn
import logging
import random
import json
import torch
import yaml
import os

from src.models.tim_net import TimNet
from src.dataset import SER_Dataset
from src.models.conformer import CNN_Conformer

current_time = datetime.now()
current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

if not os.path.exists("logs"):
    os.mkdir("logs")
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/log_{current_time}.log"),
        logging.StreamHandler()
    ],
    force = True
)

class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cpu" if not torch.cuda.is_available() else config["device"]
        self.set_random_state(int(config["seed"]))
        
        with open(self.config["data_config"]) as f:
            self.data_config = yaml.load(f, Loader=SafeLoader)
            
        json_obj = json.dumps(self.data_config, indent=4, ensure_ascii=False)
        print("data config: ")
        print(json_obj)
        
        json_obj = json.dumps(self.config, indent=4, ensure_ascii=False)
        print("general config: ")
        print(json_obj)

        self.prepare_diretories_and_logger()
        self.cre_loss = torch.nn.CrossEntropyLoss()
        
        if config["mode"] == "train":
            if config["data_config"].endswith("ravdess.yaml"):
                train_features, train_masks, train_labels = utils.load_data(
                    path=self.data_config["train_path"]
                )

                train_features, valid_features, train_masks, valid_masks, train_labels, valid_labels = train_test_split(
                    train_features, train_masks, train_labels, test_size=self.data_config["valid_size"], random_state=self.config["random_seed"]
                )
                train_features, test_features, train_masks, test_masks, train_labels, test_labels = train_test_split(
                    train_features, train_masks, train_labels, test_size=self.data_config["test_size"], random_state=self.config["random_seed"]
                )
                self.train_dl = self.prepare_dataloader(train_features, train_labels, train_masks, mode="train")
                self.val_dl = self.prepare_dataloader(valid_features, valid_labels, valid_masks, mode="test")
                self.test_dl = self.prepare_dataloader(test_features, test_labels, test_masks, mode="test")
                    
            elif config["data_config"].endswith("multilingual.yaml"):
                train_df = utils.prepare_data(
                    metadata_path=self.data_config["train_path"], 
                    wav_path=self.data_config["wavs_path"]
                )
                test_df = utils.prepare_data(
                    metadata_path=self.data_config["test_path"],
                    wav_path=self.data_config["wavs_path"]
                )
                
                train_df, val_df = train_test_split(
                    train_df, test_size=self.data_config["valid_size"], random_state=self.config["random_seed"]
                )
                
                self.train_dl = self.prepare_multilingual_dataloader(df=train_df.reset_index(), config=self.data_config, mode="train")
                self.val_dl = self.prepare_multilingual_dataloader(df=val_df.reset_index(), config=self.data_config, mode="test")
                self.test_dl = self.prepare_multilingual_dataloader(df=test_df.reset_index(), config=self.data_config, mode="test")
                
                print(f"train size: {len(train_df)}")
                print(f"val size: {len(val_df)} ")
                print(f"test size: {len(test_df)}")
            
            
                    
        elif config["mode"] == "infer":
            pass
        
        model = self.init_model()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"num params: {params}")
        
    def init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)

    def set_random_state(self, seed):
        print(f'set random_seed = {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        
    def prepare_diretories_and_logger(self):
        current_time = datetime.now()
        current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

        log_dir = f"{self.config['log_dir']}/{current_time}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"logging into {log_dir}")
            
        checkpoint_dir = self.config["checkpoint_dir"]
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            print(f'mkdir {checkpoint_dir}')
        
        self.writer = SummaryWriter(
            log_dir=log_dir
        )
        
    def init_model(self):
        with open(self.config["model_config"]) as f:
            model_config = yaml.load(f, Loader=SafeLoader)
        self.model_config = model_config
        
        if "light_ser_cnn" in self.config["model_config"]:
            pass
            # model = Light_SER(self.model_config).to(self.device)
        elif "tim_net" in self.config["model_config"]:
            model = TimNet(
                n_filters=self.data_config["hidden_dim"],
                n_label=len(self.data_config["label"].keys())).to(self.device)
            print(self.data_config["label"].keys())
        elif "cnn_transformer" in self.config["model_config"]:
            pass
            # model = CNN_Transformer().to(self.device)
            
        elif "conformer" in self.config["model_config"]:
            model = CNN_Conformer(
                self.model_config, 
                n_label=len(self.data_config["label"].keys())).to(self.device)
        
        return model
            
    def init_optimizer(self, model):
        optimizer = Adam(
            params=model.parameters(),
            betas=(self.config["beta1"], self.config["beta2"]),
            lr=self.config["lr"],
            weight_decay=float(self.config["weight_decay"]))
        
        scheduler = lr_scheduler.LinearLR(
            optimizer, 
            start_factor=self.config["start_factor"], 
            end_factor=self.config["end_factor"], 
            total_iters=self.config["total_iters"]
            )

        
        return optimizer, scheduler
    
    def prepare_ravdess_dataloader(self, features, labels, masks, mode="train"):
        dataset = SER_Dataset(
            features, labels, masks, mode=mode)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_worker"]),
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            drop_last=False,
            shuffle=True)
        
        return dataloader
    
    def prepare_multilingual_dataloader(self, df, config, mode="train"):
        dataset = SER_Dataset(
            df, config, mode=mode)
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_worker"]),
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            drop_last=False,
            shuffle=True)
        
        return dataloader
    
    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location="cpu")
        
        model_state_dict = state_dict["model_state_dict"]
        optim_state_dict = state_dict["optim_state_dict"]
        step = state_dict["step"]
        best_result = state_dict["best_result"]

        print(f'load checkpoint from {path}')        
        
        return {
            "model_state_dict":model_state_dict,
            "optim_state_dict":optim_state_dict,
            "step":step,
            "best_result":best_result
        }
    def train(self):
        print("########## start training #########")
        print("################# init model ##################")
        model = self.init_model()
        print("############### init optimizer #################")
        optimizer, scheduler = self.init_optimizer(model)
        
        step = 0
        best_wa = -1
        if os.path.exists(self.config["resume_ckpt"]):
            state_dict = self.load_checkpoint(self.config["resume_ckpt"])
            
            model.load_state_dict(state_dict["model_state_dict"])
            optimizer.load_state_dict(state_dict["optim_state_dict"])
            step = state_dict["step"]
            best_wa = state_dict["best_result"]
            
            print("load checkpoint from: ", self.config["resume_ckpt"])
        model.train()
        
        for epoch in range(int(self.config["n_epoch"])):
            train_losses, valid_losses = [], []
            _train_tqdm = tqdm(self.train_dl, desc=f"Epoch={epoch}")
            for i, batch in enumerate(_train_tqdm):
                optimizer.zero_grad()
                
                inputs = batch["inputs"].float().to(self.device)
                labels = batch["labels"].float().to(self.device)
                masks = batch["masks"].bool().to(self.device)
                
                _, preds = model(inputs=inputs, masks=masks)
                
                loss = self.cre_loss(preds, labels)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                
                train_losses.append(loss.item())
                optimizer.step()
                
                step += 1
                
                _train_tqdm.set_postfix(
                    {
                        "loss":loss.item(),
                        "lr":optimizer.param_groups[0]["lr"]
                        }
                )
            # scheduler.step()
        
            if (epoch+1) % int(self.config["evaluate_per_epoch"])==0:
                train_loss = np.mean(train_losses)
                target_names = list(self.data_config["label"].keys())
                
                model.eval()
                print(f"start validation (epoch={epoch}): ")
                valid_results = self.evaluate(val_dl=self.val_dl, model=model)
                valid_cls_result = classification_report(
                    y_pred=valid_results["predicts"], 
                    y_true=valid_results["labels"],
                    output_dict=False, zero_division=0,
                    target_names=target_names)
                
                print(f"validation result (epoch={epoch}): \n {valid_cls_result}")
                
                print(f"start testing (epoch={epoch}): ")
                test_results = self.evaluate(val_dl=self.test_dl, model=model)
                test_cls_result = classification_report(
                    y_pred=test_results["predicts"], 
                    y_true=test_results["labels"],
                    output_dict=False, zero_division=0,
                    target_names=target_names)
                
                print(f"test result (epoch={epoch}): \n{test_cls_result}")                
                test_cls_result = classification_report(
                    y_pred=test_results["predicts"], 
                    y_true=test_results["labels"],
                    output_dict=True, zero_division=0,
                    target_names=target_names)
                   
                valid_cls_result = classification_report(
                    y_pred=valid_results["predicts"], 
                    y_true=valid_results["labels"],
                    output_dict=True, zero_division=0,
                    target_names=target_names)  
                                    
                if best_wa < valid_cls_result["weighted avg"]["recall"]:
                    best_wa = valid_cls_result["weighted avg"]["f1-score"]
                    path = f'{self.config["checkpoint_dir"]}/best_war_checkpoint.pt'
                    self.save_checkpoint(path, model=model, optimizer=optimizer,best_result= best_wa, epoch=epoch, loss=train_loss, step=step)
                    print(f"test with current best checkpoint (epoch={epoch}): ")
                    self.test(checkpoint=path,test_dl=self.test_dl)                      
                model.train()
                print("############################################")
                
                self.writer.add_scalars(
                    "Loss", {
                        "train":train_loss,
                        "val":valid_results["loss"].tolist()
                    }, global_step=step
                )
                
                self.writer.add_scalars(
                    "WA", {
                        "best":best_wa,
                        "val":valid_cls_result["weighted avg"]["f1-score"],
                        "test":test_cls_result["weighted avg"]["f1-score"]
                    }, global_step=step
                )
                
                json_obj = json.dumps({
                    "weighted_avg":best_wa, 
                    "epoch":epoch, 
                    "valid_loss":valid_results["loss"].tolist(),
                    "train_loss":train_loss
                    }, indent=4, ensure_ascii=False)
                
                message = "validation result: \n" + json_obj
                print(message)
                print("############################################")
                                 
    def save_checkpoint(self, path, model, best_result,optimizer, epoch, loss, step):
        state_dict = {
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss":loss,
            "epoch":epoch,
            "step":step,
            "best_result":best_result
        }
        torch.save(state_dict, path)

    def evaluate(self, model, val_dl, mode="test"):
        predicts, labels = [], []
        
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(val_dl):
                inputs = batch["inputs"].float().to(self.device)
                _labels = batch["labels"].float().to(self.device)
                masks = batch["masks"].bool().to(self.device)
                
                _, preds = model(inputs=inputs, masks=masks)
                
                loss = self.cre_loss(preds, _labels)
                
                preds = torch.nn.functional.softmax(preds, dim=-1)
                preds = torch.argmax(preds, dim=-1)
                _labels = _labels.argmax(dim=-1)
                
                labels += _labels.cpu().tolist()
                predicts += preds.cpu().tolist()
                
                losses.append(loss.item())        
        return {
            "loss":torch.tensor(losses).mean(),
            "predicts":np.array(predicts),
            "labels":np.array(labels),
        }
    def test(self, checkpoint, test_dl):        
        model = self.init_model()            
        state_dict = self.load_checkpoint(checkpoint)
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        target_names = list(self.data_config["label"].keys())
        
        with torch.no_grad():
            test_results = self.evaluate(val_dl=test_dl, model=model)
        
        test_cls_result = classification_report(
            y_pred=test_results["predicts"], 
            y_true=test_results["labels"],
            output_dict=False, zero_division=0,
            target_names=target_names)
        
        print(test_cls_result)
        
        test_cls_result = classification_report(
            y_pred=test_results["predicts"], 
            y_true=test_results["labels"],
            output_dict=True, zero_division=0,
            target_names=target_names)
                
        return {
            "acc":test_cls_result["accuracy"],
            "war":test_cls_result["weighted avg"]["recall"],
            "uwar":test_cls_result["macro avg"]["recall"],
        }