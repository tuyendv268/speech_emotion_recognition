from src.trainer import Trainer
import yaml
from yaml.loader import SafeLoader

if __name__ == "__main__":
    with open("configs/general_config.yaml", "r") as f:
        general_config = yaml.load(f, SafeLoader)

    trainer = Trainer(general_config)
    trainer.train()