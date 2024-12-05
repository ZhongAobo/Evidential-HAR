def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"), ("18", "27"), ("20", "5"), ("24", "8"), ("28", "27"), ("30", "20")]
        # self.scenarios = [("6", "23")]

        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128
        
        
class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        # self.scenarios = [("6", "5")]
        self.scenarios = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")]

        # self.scenarios = [("0", "12")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        # self.scenarios = [("33", "12")]
        self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32")]

        # self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
                        #   ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32")]

        # self.scenarios = [("1", "1"), ("2", "1"), ("3", "1"), ("4", "1"), ("5", "1"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "1"), ("10", "1")
        #                  ,("11", "1"), ("12", "1"), ("13", "1"), ("14", "1"), ("15", "1"), ("16", "1"), ("17", "1"), ("18", "1"), ("19", "1"), ("20", "1")
        #                  ,("21", "1"), ("22", "1"), ("23", "1"), ("24", "1"), ("25", "1"), ("26", "1"), ("27", "1"), ("28", "1"), ("29", "1"), ("30", "1")
        #                  ,("31", "1"), ("32", "1"), ("33", "1"), ("34", "1"), ("35", "1")]
        # self.scenarios = [("12", "1"),("12", "2"),("12", "3"),("12", "4"),("12", "5"),("12", "6")
        #                  ,("12", "7")            ,("12", "9"),("12", "10"),("12", "11"),("12", "12")
        #                  ,("12", "13"),("12", "14"),("12", "15"),("12", "16"),("12", "17"),("12", "18")
        #                  ,("12", "19"),("12", "20"),("12", "21"),("12", "22"),("12", "23"),("12", "24")
        #                  ,("12", "25"),("12", "26"),("12", "27"),("12", "28"),("12", "29"),("12", "30")
        #                  ,("12", "31"),("12", "32"),("12", "33"),("12", "34"),("12", "35")]
        # self.scenarios = [("1", "1"),("2", "2"),("3", "3"),("4", "4"),("5", "5"),("6", "6")
        #                  ,("7", "7")            ,("9", "9"),("10", "10"),("11", "11"),("12", "12")
        #                  ,("13", "13"),("14", "14"),("15", "15"),("16", "16"),("17", "17"),("18", "18")
        #                  ,("19", "19"),("20", "20"),("21", "21"),("22", "22"),("23", "23"),("24", "24")
        #                  ,("25", "25"),("26", "26"),("27", "27"),("28", "28"),("29", "29"),("30", "30")
        #                  ,("31", "31"),("32", "32"),("33", "33"),("34", "34"),("35", "35")]
        # self.scenarios = [("35", "31")]
        self.num_classes = 6
        self.shuffle = True
        # self.drop_last = False
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR, self).__init__()
        self.sequence_len = 128
        self.scenarios = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"), ("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2")]

        # self.scenarios = [("4", "6")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        
        
class FD(object):
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        # self.scenarios = [("0", "3")]

        self.scenarios = [("0", "1"), ("0", "3"), ("1", "0"), ("1", "2"),("1", "3"),("2", "1"),("2", "3"),  ("3", "0"), ("3", "1"), ("3", "2")]
        # self.scenarios = [("0", "3")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
