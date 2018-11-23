class Config:
    def __init__(self):

        # Image specific configurations
        self.min_img_size = 600

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # RPN specific configurations
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_stride = 16
        self.rpm_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # Input specific configurations
        self.simple_label_filepath = 'C:\\Users\\pransen.ORADEV\\Desktop\\dataset\\labels.txt'
        self.config_save_file = 'config.pickle'

        # training related settings
        self.num_epochs = 3000

