import argparse
parser = argparse.ArgumentParser()



# ========  Experiments Name ================
parser.add_argument('--phase',                  default='train',         type=str, help='train or test')
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--exp_name',               default='EXP1',         type=str, help='experiment name')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='MCD',               type=str, help='NO_ADAPT, Deep_Coral, MMDA, DANN, CDAN, DIRT, DSAN, HoMM, CoDATS, AdvSKM, SASA, CoTMix, TARGET_ONLY')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'/media/xxxy/Data1/ZXY/zxy/eTime/dataset',                  type=str, help='Path containing datase2t')
parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR - FD)')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default= "cuda:0",                   type=str, help='cpu or cuda')

parser.add_argument('--dist', default='none', type=str)#######################################
parser.add_argument('--loss_type', default='IEDL', type=str)#######################################
parser.add_argument('--alpha', default=1.0, type=float)#######################################
parser.add_argument('--step_size', default=1.0, type=float)#######################################
parser.add_argument('--lr_decay', default=1.0, type=float)#######################################



parser.add_argument('--show_evi', default=False)#######################################




args = parser.parse_args()