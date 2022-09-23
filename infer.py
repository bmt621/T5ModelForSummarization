from inference import t5inference
import hydra
from src.utils import get_logger
from omegaconf import DictConfig

@hydra.main(config_path = './src/conf',config_name='config')
def main(cfg: DictConfig):
    logger = get_logger('logs.log')

    t5_infer=t5inference(cfg,logger)
    
    prompt(t5_infer)

    check = True

    while check:
        get = str(input("do you want to summarize again? (y/n) "))
        
        if get == 'y':
            prompt(t5_infer)

        elif get == 'n':
            check=False

        else:
            print("invalid response, accepts only (y/n).")


def prompt(infer):
    text = str(input("enter text to summarize: "))
    output = infer.infer_single(text)
    print("Summarized text: ",output)


if __name__ == "__main__":
    main()
