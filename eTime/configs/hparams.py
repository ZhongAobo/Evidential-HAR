## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            # 'NO_ADAPT': {'learning_rate': 0.18, 'src_cls_loss_wt': 1, 'alpha': 5.0},
            'NO_ADAPT': {'learning_rate': 0.0005, 'src_cls_loss_wt': 1, 'alpha': 5.0},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1, 'alpha': 5.0},
            "SASA": {
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001,
                "alpha": 2.0
            },
            "DDC": {
                "learning_rate": 0.001,
                "mmd_wt": 3.7991920933520342,
                "src_cls_loss_wt": 6.286301875125623,
                "domain_loss_wt": 6.36,
                "weight_decay": 0.0001,
                "alpha": 20.0
            },
            "CoDATS": {
                "domain_loss_wt": 3.2750474868706925,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.335109786953256,
                "weight_decay": 0.0001,
                "alpha": 20.0
            },
            "DANN": {
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001,
                "alpha": 5.0
            },
            "DIRT": {
                "cond_ent_wt": 0.79,
                "domain_loss_wt": 4.51,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 7.00,
                "vat_loss_wt": 9.31,
                "weight_decay": 0.0001,
                "alpha": 0.321
            },
            "DSAN": {
                "learning_rate": 0.001,
                "mmd_wt": 2.0872340713147786,
                "src_cls_loss_wt": 1.8744909939900247,
                "domain_loss_wt": 1.59,
                "weight_decay": 0.0001,
                "alpha": 0.03
            },
            "MMDA": {
                "cond_ent_wt": 1.383002023133561,
                "coral_wt": 8.36810764913737,
                "learning_rate": 0.001,
                "mmd_wt": 3.964042918489996,
                "src_cls_loss_wt": 6.794522068759213,
                "weight_decay": 0.0001,
                "alpha": 20.0
            },
            "Deep_Coral": {
                "coral_wt": 4.23035475456397,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.1013209750429822,
                "weight_decay": 0.0001,
                "alpha": 10.0
            },
            "CDAN": {
                # "cond_ent_wt": 1.2920143348777362,
                "cond_ent_wt": 1.00000000000000,
                # "domain_loss_wt": 9.545761950873414,
                "domain_loss_wt": 7.5000000000000,
                "learning_rate": 0.001,
                # "src_cls_loss_wt": 9.430292987535724,
                "src_cls_loss_wt": 7.00000000000000,
                "weight_decay": 0.0001,
                "alpha": 8.3
            },
            "AdvSKM": {
                "domain_loss_wt": 1.338788378230754,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 2.468525942065072,
                "weight_decay": 0.0001,
                "alpha": 5.0
            },
            "HoMM": {
                "hommd_wt": 2.8305712579412683,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.1282520874653523,
                "domain_loss_wt": 9.13,
                "weight_decay": 0.0001,
                "alpha": 3.0
            },
            'CoTMix': {'learning_rate': 0.001, 'mix_ratio': 0.9, 'temporal_shift': 14,
                       'src_cls_weight': 0.78, 'src_supCon_weight': 0.1, 'trg_cont_weight': 0.1,
                       'trg_entropy_weight': 0.05, "alpha": 5.0},
            'MCD': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 5.43, "alpha": 5.0},

        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'step_size': 50,##########################
            'lr_decay': 0.5#########################

        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "SASA": {
                "domain_loss_wt": 5.8045319155819515,
                "learning_rate": 3e-3,
                # "learning_rate": 0.005,
                "src_cls_loss_wt": 4.438490884851632,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "domain_loss_wt": 0.3551260369189456,
                "learning_rate": 7e-3,
                # "learning_rate": 0.005,
                "src_cls_loss_wt": 1.2534327517723889,
                "weight_decay": 0.0001
            },
            "AdvSKM": {
                "domain_loss_wt": 5.600818539370264,
                "learning_rate": 1.3e-2,
                # "learning_rate": 0.0005,
                "src_cls_loss_wt": 4.231231335081738,
                "weight_decay": 0.0001
            },
            "Deep_Coral": {
                "coral_wt": 9.50224286095279,
                "learning_rate": 2.1e-3,
                # "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.8149666724969482,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 0.27634197975549135,
                "learning_rate": 9.5e-4,
                # "learning_rate": 0.0005,
                "src_cls_loss_wt": 8.441929209893459,
                "weight_decay": 0.0001
            },
            "DDC": {
                "learning_rate": 4e-3,
                # "learning_rate": 0.0005,
                # "learning_rate": 0.035,
                "mmd_wt": 5.900770246907044,
                "src_cls_loss_wt": 1.979307877348751,
                "domain_loss_wt": 8.923,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.7021814402136783,
                "domain_loss_wt": 1.6488583075821344,
                "learning_rate": 1.1e-2,
                # "learning_rate": 0.01,
                "src_cls_loss_wt": 6.427127521674593,
                "vat_loss_wt": 5.078600240648073,
                "weight_decay": 0.0001
            },
            "MMDA": {
                "cond_ent_wt": 9.177841626283191,
                "coral_wt": 2.768290045896212,
                "learning_rate": 1.5e-3,
                # "learning_rate": 0.0005,
                "mmd_wt": 2.25231504738171,
                "src_cls_loss_wt": 8.64418208100774,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 5e-3,
                # "learning_rate": 0.001,
                "mmd_wt": 5.01196798268099,
                "src_cls_loss_wt": 7.774381653453339,
                "domain_loss_wt": 6.708,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 3.843851397373747,
                "learning_rate": 4e-3,
                # "learning_rate": 0.001,
                "src_cls_loss_wt": 1.8311375304849091,
                "domain_loss_wt": 1.102,
                "weight_decay": 0.0001
            },
            "CDAN": {
                "cond_ent_wt": 0.7559091229767906,
                "domain_loss_wt": 0.17693531166083065,
                "learning_rate": 9e-3,
                # "learning_rate": 0.0005,
                "src_cls_loss_wt": 7.764624556216286,
                "weight_decay": 0.0001
            },
            'CoTMix': {'learning_rate': 0.001, 'mix_ratio': 0.79, 'temporal_shift': 300,
                       'src_cls_weight': 0.96, 'src_supCon_weight': 0.1, 'trg_cont_weight': 0.1,
                       'trg_entropy_weight': 0.05}

        }



class WISDM():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 10,##########################
            'lr_decay': 0.9#########################

        }
        self.alg_hparams = {
            # 'NO_ADAPT': {'learning_rate': 0.008, 'src_cls_loss_wt': 1.0, 'alpha': 5.0},
            'NO_ADAPT': {'learning_rate': 0.001, 'src_cls_loss_wt': 1.0, 'alpha': 5.0},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1, 'alpha': 5.0},
            "SASA": {
                "domain_loss_wt": 1.2632988839197083,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.898676755625807,
                "weight_decay": 0.0001,
                "alpha": 7.0
            },
            "CDAN": {
                "cond_ent_wt": 0.837129024245748,
                "domain_loss_wt": 5.9197207530729266,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.983963629299826,
                "weight_decay": 0.0001,
                "alpha": 0.7
            },
            "HoMM": {
                "hommd_wt": 6.799448304230478,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 0.2563533185103576,
                "domain_loss_wt": 4.239,
                "weight_decay": 0.0001,
                "alpha": 0.5
            },
            "DANN": {
                "domain_loss_wt": 3.100000000000000,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 6.300000000000000,
                "weight_decay": 0.0001,
                "alpha": 2.76
            },
            "DIRT": {
                "cond_ent_wt": 1.6935884891647972,
                "domain_loss_wt": 7.774841143071709,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.62463958771893,
                "vat_loss_wt": 4.644539486962429,
                "weight_decay": 0.0001,
                "alpha": 1.0
            },
            "AdvSKM": {
                "domain_loss_wt": 0.17573022784621156,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 7.656694101023234,
                "weight_decay": 0.0001,
                "alpha": 0.5
            },
            "MMDA": {
                "cond_ent_wt": 7.555540424691775,
                "coral_wt": 5.254400971297628,
                "learning_rate": 0.005,
                "mmd_wt": 2.295549751091742,
                "src_cls_loss_wt": 6.653513071102565,
                "weight_decay": 0.0001,
                "alpha": 20.0
            },
            "Deep_Coral": {
                "coral_wt": 6.4881104202861755,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.66305608395703,
                "weight_decay": 0.0001,
                "alpha": 10.0
            },
            "CoDATS": {
                "domain_loss_wt": 4.574872968982744,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.860885469514424,
                "weight_decay": 0.0001,
                "alpha": 0.03
            },
            "DSAN": {
                "learning_rate": 0.005,
                "mmd_wt": 1.5468030830413808,
                "src_cls_loss_wt": 1.2981011362021273,
                "domain_loss_wt": 0.1,
                "weight_decay": 0.0001,
                "alpha": 1.0
            },
            "DDC": {
                "learning_rate": 0.001,
                "mmd_wt": 1.9901164953952095,
                "src_cls_loss_wt": 4.881899626451807,
                "domain_loss_wt": 7.595,
                "weight_decay": 0.0001,
                "alpha": 7.0
            },
            "CoTMix": {
                'learning_rate': 0.001, 
                'mix_ratio': 0.72, 
                'temporal_shift': 14,
                'src_cls_weight': 0.98, 
                'src_supCon_weight': 0.1, 
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.05,
                "alpha": 5.0}
        }


class HHAR():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 30,###############################
            'lr_decay': 0.2###############################
        }
        self.alg_hparams = {
            # 'NO_ADAPT': {'learning_rate': 0.002, 'src_cls_loss_wt': 1, 'alpha': 5.0},
            'NO_ADAPT': {'learning_rate': 0.0001, 'src_cls_loss_wt': 1, 'alpha': 5.0},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1, 'alpha': 5.0},

            "SASA": {
                "domain_loss_wt": 5.760124609738364,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.130742585941761,
                "weight_decay": 0.0001,
                "alpha": 0.5
            },
            "DSAN": {
                "learning_rate": 0.0005,
                "mmd_wt": 0.5993593617252002,
                "src_cls_loss_wt": 0.340000000000000,
                "domain_loss_wt": 0.110000000000000,
                "weight_decay": 0.0001,
                "alpha": 0.01
            },
            "CoDATS": {
                "domain_loss_wt": 9.314114040099962,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 7.700018679383289,
                "weight_decay": 0.0001,
                "alpha": 3.0
            },
            "HoMM": {
                "hommd_wt": 7.172430927893522,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.20121211752349172,
                "domain_loss_wt": 0.9824,
                "weight_decay": 0.0001,
                "alpha": 1.0
            },
            "DIRT": {
                "cond_ent_wt": 1.329734510542011,
                "domain_loss_wt": 6.632293308809388,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 7.729881324550688,
                "vat_loss_wt": 6.912258476982827,
                "weight_decay": 0.0001,
                "alpha": 2.0
            },
            "AdvSKM": {
                "domain_loss_wt": 1.8649335076712072,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 3.961611563054495,
                "weight_decay": 0.0001,
                "alpha": 10.0
            },
            "DDC": {
                "learning_rate": 0.0005,
                "mmd_wt": 8.355791702302787,
                "src_cls_loss_wt": 1.2079058664226126,
                "domain_loss_wt": 0.2048,
                "weight_decay": 0.0001,
                "alpha": 20.0
            },
            "CDAN": {
                "cond_ent_wt": 0.2000000000000000,
                "domain_loss_wt": 2.0000000000000000,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 3.1000000000000000,
                "weight_decay": 0.0001,
                "alpha": 0.3
            },
            "DANN": {
                "domain_loss_wt": 1.0296390274908802,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 2.038458138479581,
                "weight_decay": 0.0001,
                "alpha": 5.0
            },
            "Deep_Coral": {
                "coral_wt": 5.9357031653707475,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.43859323168654,
                "weight_decay": 0.0001,
                "alpha": 0.1
            },
            "MMDA": {
                "cond_ent_wt": 6.707871745810609,
                "coral_wt": 5.903714930042433,
                "learning_rate": 0.005,
                "mmd_wt": 6.480169289397163,
                "src_cls_loss_wt": 0.18878476669902317,
                "weight_decay": 0.0001,
                "alpha": 1.0
            },
            'CoTMix': {'learning_rate': 0.001, 'mix_ratio': 0.52, 'temporal_shift': 14,
                       'src_cls_weight': 0.8, 'src_supCon_weight': 0.1, 'trg_cont_weight': 0.1,
                       'trg_entropy_weight': 0.05,"alpha": 5.0}

        }


class FD():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 10,##########################
            'lr_decay': 0.9#########################
        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "SASA": {
                "domain_loss_wt": 0.7821851095870519,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 7.680225091930735,
                "weight_decay": 0.0001
            },
            "MMDA": {
                "cond_ent_wt": 8.12868726468387,
                "coral_wt": 7.2734249221691005,
                "learning_rate": 0.0005,
                "mmd_wt": 4.967077206689191,
                "src_cls_loss_wt": 0.30259189730747005,
                "weight_decay": 0.0001
            },
            "AdvSKM": {
                "domain_loss_wt": 9.377024659182622,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 0.7569318345582794,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 6.719563315664067,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 1.5584167741262964,
                "domain_loss_wt": 0.9824,
                "weight_decay": 0.0001
            },
            "Deep_Coral": {
                "coral_wt": 7.493856538302936,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 1.452466194151791,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 4.753485587751647,
                "domain_loss_wt": 7.427507171955081,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 9.818770948448943,
                "vat_loss_wt": 9.609164719194178,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 0.005,
                "mmd_wt": 7.278792967879357,
                "src_cls_loss_wt": 2.5146121077752395,
                "domain_loss_wt": 0.16,
                "weight_decay": 0.0001
            },
            "CDAN": {
                "cond_ent_wt": 0.553637609557987,
                "domain_loss_wt": 6.759045461432962,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.854042579661701,
                "weight_decay": 0.0001
            },
            "DDC": {
                "learning_rate": 0.005,
                # "learning_rate": 0.08,
                "mmd_wt": 6.701050990813831,
                "src_cls_loss_wt": 1.1626428404763771,
                "domain_loss_wt": 0.2048,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "domain_loss_wt": 0.6990097136753354,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.57338373194037,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 5.221878412210977,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.233865748743297,
                "weight_decay": 0.0001
            },
            'CoTMix': {'learning_rate': 0.001, 'mix_ratio': 0.52, 'temporal_shift': 14,
                       'src_cls_weight': 0.8, 'src_supCon_weight': 0.1, 'trg_cont_weight': 0.1,
                       'trg_entropy_weight': 0.05}
        }