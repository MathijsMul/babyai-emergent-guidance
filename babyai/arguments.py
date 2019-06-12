"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default=None,
                            help="name of the environment to train on (REQUIRED)")
        self.add_argument("--model", default=None,
                            help="name of the model (default: ENV_ALGO_TIME)")
        self.add_argument("--pretrained-model", default=None,
                            help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
        self.add_argument("--seed", type=int, default=1,
                            help="random seed; if 0, a random random seed will be used  (default: 1)")
        self.add_argument("--task-id-seed", action='store_true',
                            help="use the task id within a Slurm job array as the seed")
        self.add_argument("--procs", type=int, default=64,
                            help="number of processes (default: 64)")
        self.add_argument("--tb", action="store_true", default=False,
                            help="log into Tensorboard")

        # Training arguments
        self.add_argument("--log-interval", type=int, default=1,
                            help="number of updates between two logs (default(Mathijs): 1, used to be 10)")
        self.add_argument("--save-interval", type=int, default=1000,
                            help="number of updates between two saves (default: 1000, 0 means no saving)")
        self.add_argument("--frames", type=int, default=int(9e10),
                            help="number of frames of training (default: 9e10)")
        self.add_argument("--patience", type=int, default=100,
                            help="patience for early stopping (default: 100)")
        self.add_argument("--epochs", type=int, default=1000000,
                            help="maximum number of epochs")
        self.add_argument("--frames-per-proc", type=int, default=40,
                            help="number of frames per process before update (default: 40)")
        self.add_argument("--lr", type=float, default=1e-4,
                            help="learning rate (default: 1e-4)")
        self.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
        self.add_argument("--recurrence", type=int, default=20,
                            help="number of timesteps gradient is backpropagated (default: 20)")
        self.add_argument("--optim-eps", type=float, default=1e-5,
                            help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--optim-alpha", type=float, default=0.99,
                            help="RMSprop optimizer apha (default: 0.99)")
        self.add_argument("--batch-size", type=int, default=1280,
                                help="batch size for PPO (default: 1280)")
        self.add_argument("--entropy-coef", type=float, default=0.01,
                            help="entropy term coefficient (default: 0.01)")
        self.add_argument("--dropout", type=float, default=0.5,
                          help="dropout probability for processed corrections (default: 0.5)")

        self.add_argument("--save-each-epoch", action="store_true", default=False,
                          help="store model at each epoch")
        self.add_argument("--class-weights", action="store_true", default=False,
                          help="use class weights in loss function")
        self.add_argument("--compute-cic", action="store_true", default=False,
                          help="compute and log causal influence of communication metric after each epoch")

        # Model parameters
        self.add_argument("--image-dim", type=int, default=128,
                            help="dimensionality of the image embedding")
        self.add_argument("--memory-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--instr-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--no-instr", action="store_true", default=False,
                            help="don't use instructions in the model")
        self.add_argument("--instr-arch", default="gru",
                            help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
        self.add_argument("--no-mem", action="store_true", default=False,
                            help="don't use memory in the model")
        self.add_argument("--arch", default='expert_filmcnn',
                            help="image embedding architecture")
        self.add_argument("--learner", action="store_true", default=False,
                          help="use ordinary learner")

        # Corrector parameters
        self.add_argument("--corrector", action="store_true", default=False,
                          help="use correction module")
        self.add_argument("--corr-length", type=int, default=2,
                          help="length of correction messages (max length if --var-corr-length true)")
        self.add_argument("--corr-own-vocab", action="store_true", default=False,
                          help="corrector uses its own vocabulary instead of instruction vocabulary")
        self.add_argument("--corr-embedding-dim", type=int, default=0,
                            help="embedding dimensionality for corrector")
        self.add_argument("--corr-vocab-size", type=int, default=3,
                            help="vocabulary size of corrector")
        self.add_argument("--pretrained-corrector", type=str, default=None,
                          help="location of pretrained corrector to use and freeze")
        self.add_argument("--show-corrections", action="store_true", default=False,
                            help="show correction messages")
        self.add_argument("--corrector-frozen", action="store_true", default=False,
                          help="freeze pretrained corrector")
        self.add_argument("--random-corrector", action="store_true", default=False,
                          help="randomize correction messages")
        self.add_argument("--var-corr-length", action="store_true", default=False,
                          help="variable length correction messages with penalty for longer ones")
        self.add_argument("--corr-loss-coef", type=float, default=0.1,
                          help="correction loss coefficient (untested default: 0.1)")
        self.add_argument("--weigh-corrections", action="store_true", default=False,
                          help="weigh corrections depending on entropy of previous timestep")
        self.add_argument("--correction-weight-loss-coef", type=float, default=1.0,
                          help="coefficient for correction weight loss")

        # Validation parameters
        self.add_argument("--val-seed", type=int, default=0,
                            help="seed for environment used for validation (default: 0)")
        self.add_argument("--val-interval", type=int, default=1,
                            help="number of epochs between two validation checks (default: 1)")
        self.add_argument("--val-episodes", type=int, default=500,
                            help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")

    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args()

        # Set seed for all randomness sources
        if args.seed == 0:
            args.seed = np.random.randint(10000)
        if args.task_id_seed:
            args.seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
            print('set seed to {}'.format(args.seed))

        # TODO: more validation

        return args
