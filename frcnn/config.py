class Config:
    def __init__(self):

        # Image specific configurations
        self.img_size = 600

        # RPN specific configurations
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_stride = 16
        self.rpm_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # Input specific configurations
        self.simple_label_file = 'labels.txt'
        self.config_save_file = 'config.pickle'

