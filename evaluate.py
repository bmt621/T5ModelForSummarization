from src.T5evaluator import T5Evaluator
import hydra
from omegaconf import DictConfig
from src.utils import get_logger



@hydra.main(config_path='./src/conf',config_name='config')
def main(cfg:DictConfig):
    logger = get_logger('logs.log')
    evaluator = T5Evaluator(cfg,logger)

    evaluator.evaluate(batch_size = cfg.training_configs['eval_batch_size'])



if __name__ == "__main__":
    main()


