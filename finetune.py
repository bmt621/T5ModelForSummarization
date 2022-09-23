from asyncio.log import logger
from distutils.log import Log
from src.utils import *
from src.T5Trainer import t5Trainer
import hydra
from omegaconf import DictConfig


@hydra.main(config_path='./src/conf',config_name="config")
def main(cfg: DictConfig)->None:
    
    logger_path = "C:/Users/USER/Desktop/T5ForSummarization/logs.log"
    logger = get_logger(logger_path)

    trainer = t5Trainer(cfg,logger)

    trainer.train()
    

if __name__ == "__main__":
    main()    
    