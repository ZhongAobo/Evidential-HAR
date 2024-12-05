from trainers.train import Trainer
from trainers import args
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    # create trainier object
    trainer = Trainer(args)

    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()



#TODO:
# 1- Change the naming of the functions ---> ( Done)
# 2- Change the algorithms following DCORAL --> (Done)
# 3- Keep one trainer for both train and test -->(Done)
# 4- Create the new joint loader that consider the all possible batches --> Done
# 5- Implement Lower/Upper Bound Approach --> Done
# 6- Add the best hparams --> Done
# 7- Add pretrain based methods (ADDA, MCD, MDD)
