import math
import time
import importlib
import sys
import os
import cv2
from jsonargparse import ArgumentParser, ActionParser
import numpy as np
import copy

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn

# TrTr
import datasets.utils
from datasets.utils import crop_hwc, crop_image, siamfc_like_scale, get_exemplar_size, get_context_amount # TODO: move to utils
from external_tracker import build_external_tracker
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from models import build_model
from models.backbone import Backbone as Resnet
from models.tracker import get_args_parser as baseline_tracker_args_parser

# External Module (TODO: remove)
sys.path.append(os.path.join(os.path.dirname(__file__), '../external_tracker/external_module/pytracking'))
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features import augmentation
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from pytracking.tracker.atom.optim import ConvProblem, FactorizedConvProblem


class Tracker():

    def __init__(self, model, postprocess, search_size, dcf_params, postprocess_params):

        dcf_param_module = importlib.import_module('pytracking.parameter.atom.default_vot')
        self.dcf_params = dcf_param_module.parameters()

        # TrTr model
        self.model = model
        self.model.eval()
        self.postprocess = postprocess

        self.search_size = search_size
        backbone_stride = model.backbone.stride
        self.heatmap_size = (self.search_size + backbone_stride - 1) // backbone_stride
        self.size_lpf = postprocess_params.tracking_size_lpf
        self.size_penalty_k = postprocess_params.tracking_size_penalty_k


        hanning = np.hanning(self.heatmap_size)
        self.window = torch.as_tensor(np.outer(hanning, hanning).flatten())
        self.window_factor = postprocess_params.window_factor
        self.score_threshold = postprocess_params.score_threshold
        self.window_steps = postprocess_params.window_steps
        self.image_normalize = T.Compose([
            T.ToTensor(), # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.first_frame = False

        # Online DCF
        self.dcf_layers = dcf_params.layers
        self.dcf_rate =  dcf_params.rate
        self.dcf_feature_sz = dcf_params.size
        self.dcf_fparams = TensorList(self.dcf_params.features.get_fparams('feature_params').list() * len(self.dcf_layers))
        self.dcf_params.train_skipping = 10 # TODO: tuning 10 - 20 (vot-toolkit)
        self.dcf_params.sample_memory_size  = dcf_params.sample_memory_size

        # smaller augmentation with more init samples
        self.init_training_frame_num = 1 # parameter => important factor
        if self.init_training_frame_num > 1:
            self.dcf_params.augmentation = {'fliplr': True,
                                            'rotate': [-5, 10, -30, 60],
                                            'blur': [(2, 0.2), (1, 3)],
                                            'relativeshift': [(0.6, 0.6), (-0.6, -0.6)],
                                            'dropout': (3, 0.2)}



        self.invalid_bbox_cnt = 0
        self.invalid_bbox_score_cnt = 0
        self.invalid_bbox_cnt_max = 5 # parameter
        self.invalid_bbox_score_cnt_max = 1 # parameter
        self.boundary_margin = 0.01 # parameter: pixel? (300 -> 3)
        self.boundary_target_score_threshold = 0.2 # parameter
        self.relax_size_margin = 0.05
        self.hard_size_margin = 0.01 # parameter: pixel? (300 -> 6)
        self.boundary_recovery = postprocess_params.boundary_recovery
        self.recovery_flag = False

        self.lost_target_recovery = postprocess_params.lost_target_recovery
        self.lost_target_margin = postprocess_params.lost_target.boundary_margin
        self.translation_threshold = postprocess_params.lost_target.translation_threshold
        self.lost_target_score_threshold = postprocess_params.lost_target.score_threshold
        self.lost_target_cnt_threshold = postprocess_params.lost_target.cnt_threshold
        self.last_valid_position = None
        self.lost_target_cnt = 0

        self.hard_negative_recovery = postprocess_params.hard_negative_recovery

        # false positive
        self.max_false_postive = 3
        self.relative_valid_score_threshold = 0.25
        self.absolute_valid_score_threshold = 0.66 # a fine tuning parameter based on VOT2019 balls

        # do not do false positive remover if we have to do recovery from boundary or lost target
        if self.boundary_recovery or self.lost_target_recovery:
            self.max_false_postive = 0

        # initial fast motion
        self.max_translation = get_exemplar_size() / 2  # heuristic paramter to detect fast motion: half of exemplar_size (i.e., 127)
        self.default_search_size = search_size # max for backbone of resnet50
        self.expand_search_size = 500  # max for backbone of resnet50
        self.defualt_window_factor = self.window_factor

        # recovery
        self.recovery_score_low_threshold = 0.0
        self.recovery_score_high_threshold = 0.0

    def init(self, image, bbox):

        # Get position and size
        # NOTE: if you use torch.Tensor, the default type is float32
        # https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor
        # That is the reason why it has a slight worse performance than list operation whcih is  double64 type.
        # But right now it is OK
        self.target_pos = torch.Tensor([bbox[1] + bbox[3]/2, bbox[0] + bbox[2]/2]) # center
        self.init_target_pos = self.target_pos
        self.target_sz = torch.Tensor([bbox[3], bbox[2]]) # real size in pixel

        # For restart
        self.search_size = self.default_search_size
        self.window_factor = self.defualt_window_factor # heuristic
        stride = self.model.backbone.stride
        self.heatmap_size = (self.search_size + stride - 1) // stride
        hanning = np.hanning(self.heatmap_size)
        self.window = torch.as_tensor(np.outer(hanning, hanning).flatten())


        # TrTr
        bbox_xyxy  = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        channel_avg = np.mean(image, axis=(0, 1))
        # get crop
        s_z, scale_z = siamfc_like_scale(bbox_xyxy)
        s_x = self.search_size / scale_z
        #print(s_z, s_x, scale_z, image.shape, bbox)
        template_image, _ = crop_image(image, bbox_xyxy, padding = channel_avg)

        # get mask
        ## reverse order of target_pos and target_sz := [y,x]
        self.init_template_mask = [0, 0, template_image.shape[1], template_image.shape[0]] # [x1, y1, x2, y2]
        if self.target_pos[1] < s_z/2: # x1
            self.init_template_mask[0] = (s_z/2 - self.target_pos[1]) * scale_z
        if self.target_pos[0] < s_z/2: # y1
            self.init_template_mask[1] = (s_z/2 - self.target_pos[0]) * scale_z
        if self.target_pos[1] + s_z/2 > image.shape[0]: # x2
            self.init_template_mask[2] = self.init_template_mask[2] - (self.target_pos[1] + s_z/2 - image.shape[1]) * scale_z
        if self.target_pos[1] + s_z/2 > image.shape[1]: # y2
            self.init_template_mask[3] = self.init_template_mask[3] - (self.target_pos[0] + s_z/2 - image.shape[0]) * scale_z
        # normalize and conver to torch.tensor
        self.init_template = self.image_normalize(np.round(template_image).astype(np.uint8)).cuda()
        self.first_frame = True

        self.init_target_sz = self.target_sz
        self.init_s_x = s_x

        # reset recovery
        self.recovery_flag = False
        self.invalid_bbox_cnt = 0
        self.invalid_bbox_score_cnt = 0
        self.lost_target_cnt = 0
        self.last_valid_position = None

        # dcf initialize
        self.dcf_frame_num = 1
        self.init_training_images = []
        self.init_training_target_pos = []
        self.init_training_target_sz = []
        self.init_training_target_scale = []
        self.init_training_image_channel_avgs = []

        self.init_trtr_score = None
        self.init_dcf_score = None

        self.init_training_images.append(copy.deepcopy(image))
        self.init_training_target_pos.append(copy.deepcopy(self.target_pos))
        self.init_training_target_sz.append(copy.deepcopy(self.target_sz))
        self.init_training_image_channel_avgs.append(channel_avg)
        self.init_training_target_scale.append(1 / scale_z)
        self.dcf_target_scale = 1/ scale_z

        if self.dcf_frame_num == self.init_training_frame_num:
            # do initialize training for online DCF
            self.dcf_init()

        # for visualize
        debug_bbox = torch.round(box_cxcywh_to_xyxy(torch.cat([torch.tensor([63.5, 63.5]),  torch.Tensor([bbox[2], bbox[3]]) * scale_z]))).int()
        debug_image = cv2.rectangle(template_image,
                                    (debug_bbox[0], debug_bbox[1]),
                                    (debug_bbox[2], debug_bbox[3]),(0,255,0),3)

        return {'template_image': debug_image}

    def dcf_init(self):

        assert len(self.init_training_images) == len(self.init_training_target_pos) == len(self.init_training_image_channel_avgs) == len(self.init_training_target_sz) == len(self.init_training_target_scale) == self.init_training_frame_num

        # Initialize some stuff
        if not self.dcf_params.has('device'):
            self.dcf_params.device = 'cuda' if self.dcf_params.use_gpu else 'cpu'

        # Set search area
        self.dcf_img_sample_sz = self.search_size * torch.ones(2)

        # Set sizes
        self.dcf_img_support_sz = self.dcf_img_sample_sz
        if self.dcf_feature_sz == 0:
            self.dcf_feature_sz = self.heatmap_size
            if self.model.backbone.dilation:
                self.dcf_feature_sz = self.dcf_feature_sz // 2
        self.dcf_feature_sz_list = TensorList([torch.as_tensor([self.dcf_feature_sz, self.dcf_feature_sz])] * len(self.dcf_layers))
        self.dcf_output_sz = self.dcf_img_support_sz # use fourier to get same size with img_support_sz
        self.dcf_kernel_size = self.dcf_fparams.attribute('kernel_size')[0]
        self.dcf_compressed_dim = self.dcf_fparams.attribute('compressed_dim', None)[0]
        #print(self.dcf_img_support_sz, self.dcf_feature_sz_list, self.dcf_output_sz, self.dcf_kernel_size, self.dcf_compressed_dim)

        # Optimization options
        self.dcf_output_window = dcf.hann2d(self.dcf_output_sz.long(), centered=False).to(self.dcf_params.device)

        # Initialize some learning things
        self.dcf_init_learning()

        # Setup scale bounds
        all_train_x = None
        all_init_y = None
        for im_id, (image, target_pos, target_sz, target_scale, channel_avg) in enumerate(zip(self.init_training_images, self.init_training_target_pos, self.init_training_target_sz, self.init_training_target_scale, self.init_training_image_channel_avgs)):

            # if im_id == 0:
            #     continue

            # Extract and transform sample
            train_x = self.dcf_generate_init_samples(image, target_pos, target_scale, channel_avg)

            # Generate label function
            init_y = self.dcf_init_label_function(train_x, target_pos, target_sz, target_scale)


            if all_train_x is None:
                all_train_x = train_x
            else:
                for idx, x in enumerate(train_x):
                    all_train_x[idx] = torch.cat([all_train_x[idx], x], 0)
            if all_init_y is None:
                all_init_y = init_y
            else:
                for idx, y in enumerate(init_y):
                    all_init_y[idx] = torch.cat([all_init_y[idx], y], 0)

        # init Projectiom Matrix
        self.dcf_projection_matrix = TensorList(
            [ex.new_zeros(self.dcf_compressed_dim,ex.shape[1],1,1).normal_(0,1/math.sqrt(ex.shape[1])) for ex in all_train_x])

        # Init memory
        self.dcf_init_memory(all_train_x)

        # Init optimizer and do initial optimization
        start = time.time()
        self.dcf_init_optimization(all_train_x, all_init_y)
        # print("init optimization time: {}".format(time.time() - start))


    def dcf_init_optimization(self, train_x, init_y):
        # Initialize filter
        filter_init_method = self.dcf_params.get('filter_init_method', 'zeros')
        self.dcf_filter = TensorList(
            [x.new_zeros(1, self.dcf_compressed_dim, self.dcf_kernel_size[0], self.dcf_kernel_size[1]) for x in train_x])

        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.dcf_filter:
                f.normal_(0, 1/f.numel())
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Setup factorized joint optimization
        self.projection_reg = self.dcf_fparams.attribute('projection_reg')

        self.dcf_joint_problem = FactorizedConvProblem(self.dcf_init_training_samples, init_y, self.dcf_filter_reg,
                                                       self.projection_reg, self.dcf_params, self.dcf_init_sample_weights,
                                                       self.dcf_projection_activation, self.dcf_response_activation)

        # Variable containing both filter and projection matrix
        joint_var = self.dcf_filter.concat(self.dcf_projection_matrix)

        # Initialize optimizer
        self.dcf_joint_optimizer = GaussNewtonCG(self.dcf_joint_problem, joint_var)

        # Do joint optimization
        self.dcf_joint_optimizer.run(self.dcf_params.init_CG_iter // self.dcf_params.init_GN_iter, self.dcf_params.init_GN_iter)


        # Re-project samples with the new projection matrix
        compressed_samples = self.dcf_project_sample(self.dcf_init_training_samples, self.dcf_projection_matrix)
        for train_samp, init_samp in zip(self.dcf_training_samples, compressed_samples):
            train_samp[:init_samp.shape[0],...] = init_samp


        # Initialize optimizer
        self.dcf_conv_problem = ConvProblem(self.dcf_training_samples, self.dcf_y, self.dcf_filter_reg, self.dcf_sample_weights, self.dcf_response_activation)
        self.dcf_filter_optimizer = ConjugateGradient(self.dcf_conv_problem, self.dcf_filter, fletcher_reeves=self.dcf_params.fletcher_reeves)

        # Transfer losses from previous optimization
        self.dcf_filter_optimizer.residuals = self.dcf_joint_optimizer.residuals
        self.dcf_filter_optimizer.losses = self.dcf_joint_optimizer.losses

        # Post optimization
        self.dcf_filter_optimizer.run(self.dcf_params.post_init_CG_iter)

        # Free memory
        del self.dcf_init_training_samples
        del self.dcf_joint_problem, self.dcf_joint_optimizer

        self.init_dcf_filter = copy.deepcopy(self.dcf_filter)
        self.init_dcf_filter_optimizer_residuals = copy.deepcopy(self.dcf_filter_optimizer.residuals)
        self.init_dcf_filter_optimizer_losses = copy.deepcopy(self.dcf_filter_optimizer.losses)

    def _bbox_clip(self, bbox, boundary):
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(boundary[1]-1, bbox[2])
        y2 = min(boundary[0]-1, bbox[3])

        return [x1, y1, x2, y2]

    def track(self, image):

        # ------- online heatmap and localization ------- #
        prev_pos = self.target_pos
        prev_sz = self.target_sz

        prev_bbox_xyxy = [prev_pos[1] - prev_sz[1] / 2,
                          prev_pos[0] - prev_sz[0] / 2,
                          prev_pos[1] + prev_sz[1] / 2,
                          prev_pos[0] + prev_sz[0] / 2]
        s_z, scale_z = siamfc_like_scale(prev_bbox_xyxy)
        s_x = self.search_size / scale_z

        channel_avg = [0, 0, 0]
        if prev_pos[0] - s_x/2 <  1 or prev_pos[1] - s_x/2 <  1 or prev_pos[1] + s_x/2 > image.shape[1] - 1 or prev_pos[0] + s_x/2 > image.shape[0] - 1:
            channel_avg = [np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])] # much faster than np.mean and also fater than np.enisum

        if self.recovery_flag:
            # use
            s_x = np.max([np.min([np.max(image.shape[:2]) / 2, np.min(image.shape[:2])]), self.init_s_x])
            scale_z = self.search_size / s_x

            bbox = [prev_pos[1] - s_x / 2, prev_pos[0] - s_x / 2, prev_pos[1] + s_x / 2, prev_pos[0] + s_x / 2]
            search_image = crop_hwc(image, bbox, self.search_size, channel_avg)
        else:
            _, search_image = crop_image(image, prev_bbox_xyxy, padding = channel_avg, instance_size = self.search_size)

        # get mask
        search_mask = [0, 0, self.search_size, self.search_size]
        if prev_pos[1] < s_x/2:
            search_mask[0] = (s_x/2 - prev_pos[1]) * scale_z
        if prev_pos[0] < s_x/2:
            search_mask[1] = (s_x/2 - prev_pos[0]) * scale_z
        if prev_pos[1] + s_x/2 > image.shape[1]:
            search_mask[2] = search_mask[2] - (prev_pos[1] + s_x/2 - image.shape[1]) * scale_z
        if prev_pos[0] + s_x/2 > image.shape[0]:
            search_mask[3] = search_mask[3] - (prev_pos[0] + s_x/2 - image.shape[0]) * scale_z

        # normalize and conver to torch.tensor
        search = self.image_normalize(np.round(search_image).astype(np.uint8)).cuda()

        with torch.no_grad():
            if self.first_frame:
                outputs = self.model(nested_tensor_from_tensor_list([search], [torch.as_tensor(search_mask).float()]),
                                     nested_tensor_from_tensor_list([self.init_template], [torch.as_tensor(self.init_template_mask).float()]))
                self.first_frame = False
            else:
                outputs = self.model(nested_tensor_from_tensor_list([search], [torch.as_tensor(search_mask).float()]))

        all_features = outputs["all_features"]
        outputs = self.postprocess(outputs)


        # ------- online heatmap and localization ------- #
        self.dcf_frame_num += 1

        dcf_heatmap = None
        flag = 'initialize'

        if self.dcf_frame_num > self.init_training_frame_num:
            # Get sample
            sample_pos = self.target_pos.round()
            test_x = self.dcf_project_sample(self.dcf_feature_preprocess(all_features))

            # Compute scores
            scores_raw = self.dcf_apply_filter(test_x)
            translation_vec, s, flag = self.dcf_localize_target(scores_raw)
            dcf_heatmap = torch.clamp(s[0], min = 0)

        if self.recovery_flag:
            flag = 'recovery'

        # use the last result as template image
        if flag == 'hard_negative' and self.hard_negative_recovery and not self.recovery_flag:
            template_image, _ = crop_image(self.prev_image, prev_bbox_xyxy, padding = self.prev_channel_avg)

            # get mask
            template_mask = [0, 0, template_image.shape[1], template_image.shape[0]] # [x1, y1, x2, y2]
            if prev_pos[1] < s_z/2:
                template_mask[0] = (s_z/2 - prev_pos[1]) * scale_z
            if prev_pos[0] < s_z/2:
                template_mask[1] = (s_z/2 - prev_pos[0]) * scale_z
            if prev_pos[1] + s_z/2 > image.shape[1]:
                template_mask[2] = template_mask[2] - (prev_pos[1] + s_z/2 - image.shape[1]) * scale_z
            if prev_pos[0] + s_z/2 > image.shape[0]:
                template_mask[3] = template_mask[3] - (prev_pos[0] + s_z/2 - image.shape[0]) * scale_z

            # normalize and conver to torch.tensor
            # TODO: use extracted feature from search image
            template = self.image_normalize(np.round(template_image).astype(np.uint8)).cuda()
            with torch.no_grad():
                outputs = self.model(nested_tensor_from_tensor_list([search], [torch.as_tensor(search_mask).float()]),
                                     nested_tensor_from_tensor_list([template], [torch.as_tensor(template_mask).float()]))
                self.first_frame = True

            all_features = outputs["all_features"]
            outputs = self.postprocess(outputs)
            #print("use last result as template image")


        # Combine with TrTr tracking framework
        out = self.combine(image.shape[:2], prev_pos[[1,0]], prev_sz[[1,0]], scale_z, outputs, search_image, flag, dcf_heatmap = dcf_heatmap, test_x = test_x)

        # ------- check lost target -----#
        # we check the static target (position is almost constant) with the close range to the boundary
        # the threshould contain translation, static cnt, the score with init template
        if self.lost_target_recovery and not self.recovery_flag:
            # print('trtr score: {}; bbox_in_search_image: {}, dcf_map: {}'.format(out['trtr_score'], out['bbox_in_search_image'].numpy(), dcf_heatmap.shape))

            # TODO: also cosider the size of the bbox, and the change of the size
            if self.last_valid_position is not None:
                 translation = torch.norm(self.target_pos - self.last_valid_position).item()
                 translation_threshold = self.translation_threshold * min(image.shape[:2])
                 #print('translation: {}, translation_threshold: {}, target_pos: {}, last_valid_position: {}, image shape: {}'.format(translation, translation_threshold, self.target_pos.numpy(), self.last_valid_position.numpy(), image.shape))

                 close_boundary = (self.target_pos[1] < self.lost_target_margin * image.shape[1] or self.target_pos[1] > (1 - self.lost_target_margin ) * image.shape[1]) or (self.target_pos[0] < self.lost_target_margin * image.shape[0] or self.target_pos[0] > (1 - self.lost_target_margin ) * image.shape[0])
                 if translation < translation_threshold  and close_boundary:
                         self.lost_target_cnt += 1
                         #print("self.lost_target_cnt: ", self.lost_target_cnt)
                 else:
                     # reset since large
                     self.lost_target_cnt = 0
                     self.last_valid_position = self.target_pos
            else:
                self.last_valid_position = self.target_pos

            if self.lost_target_cnt > self.lost_target_cnt_threshold:

                dcf_heatmap = self.dcf_localize_target(operation.conv2d(test_x, self.init_dcf_filter, mode='same'))[1][0]
                dcf_heatmap /= len(self.dcf_layers)
                # TODO: should be a region of bounding box for both dcf_heatmap and trtr_heatmap, not the center point or the max score.
                dcf_score = dcf_heatmap[0][self.search_size //2][self.search_size //2]
                #print("check the score for the static target, trtr score: {} / {}, posititon in search image{}, dcf score: {} / {}, ".format(out['trtr_score'], self.init_trtr_score, out['bbox_in_search_image'], dcf_score, self.init_dcf_score))

                if out['trtr_score'] < self.lost_target_score_threshold * self.init_trtr_score  or dcf_score < self.lost_target_score_threshold * self.init_dcf_score:

                    print("reset the tracker becuase of target lost. lost cnt: {} / {}, trtr score: {} / {}, dcf score: {} / {}, ".format(self.lost_target_cnt, self.lost_target_cnt_threshold , out['trtr_score'], self.init_trtr_score, dcf_score, self.init_dcf_score))
                    # reset the dcf optimizer
                    self.dcf_filter_optimizer.residuals = self.init_dcf_filter_optimizer_residuals
                    self.dcf_filter_optimizer.losses = self.init_dcf_filter_optimizer_losses
                    self.dcf_filter = copy.deepcopy(self.init_dcf_filter)

                    self.dcf_num_stored_samples = self.dcf_num_init_samples.copy()
                    self.dcf_previous_replace_ind = [None] * len(self.dcf_num_stored_samples)
                    for train_samp, y_memory, sw, init_sw, num in zip(self.dcf_training_samples, self.dcf_y, self.dcf_sample_weights, self.dcf_init_sample_weights, self.dcf_num_init_samples):
                        train_samp[num:] = 0
                        y_memory[num:] = 0
                        sw[:num] = init_sw
                        sw[num:] = 0


                    self.target_sz = self.init_target_sz
                    self.target_pos = torch.Tensor([image.shape[0]/2, image.shape[1]/2]) # center of image

                    self.last_valid_position = None
                    self.lost_target_cnt = 0
                    self.recovery_flag = True # TODO: change to other name



        # heuristic solution:
        # if first target motion is found in the first tracking frame, we expand the search size
        if self.dcf_frame_num == 2:
            bbox_ct = out['bbox_in_search_image']
            delta = (bbox_ct - self.search_size / 2).abs().max().item()
            if delta > self.max_translation:
                # print('fast target motion in the first frame : {}/{}'.format(delta, self.max_translation))
                self.search_size = self.expand_search_size # heuristic
                self.window_factor /= 2 # heuristic

                # update baseline tracker configuration
                stride = self.model.backbone.stride
                self.heatmap_size = (self.search_size + stride - 1) // stride
                hanning = np.hanning(self.heatmap_size)
                self.window = torch.as_tensor(np.outer(hanning, hanning).flatten())

                # re training DCF
                # TODO: the projection matrix maybe not necessary. try to only training self.dcf_filter
                self.dcf_init()
                flag = 'not_found' # do not update the memory


        # ------- online DF ------- #
        if self.dcf_frame_num <= self.init_training_frame_num:

            self.init_training_images.append(image)
            self.init_training_target_pos.append(self.target_pos)
            self.init_training_target_sz.append(self.target_sz)
            self.init_training_target_scale.append(self.dcf_target_scale) # TODO:: please calculate from target_sz
            self.init_training_image_channel_avgs.append(channel_avg)

            if self.dcf_frame_num == self.init_training_frame_num:
                # do initialize training for online DCF
                print("do initialize for DCF")
                self.dcf_init()
        else:
            # Check flags and set learning rate if hard negative
            update_flag = flag not in ['not_found', 'uncertain', 'recovery']
            hard_negative = (flag == 'hard_negative')
            learning_rate = self.dcf_params.hard_negative_learning_rate if hard_negative else None

            if update_flag:
                # Get train sample
                train_x = TensorList([x[0:1, ...] for x in test_x])

                # Create label for sample
                train_y = self.dcf_get_label_function(self.target_pos, sample_pos, self.dcf_target_scale)

                # Update memory
                self.dcf_update_memory(train_x, train_y, learning_rate)

                # Update image
                self.prev_image = image
                self.prev_channel_avg = channel_avg

            # Train filter
            if hard_negative:
                start = time.time()
                self.dcf_filter_optimizer.run(self.dcf_params.hard_negative_CG_iter)
                #print("hard negative dcf updating time: {}".format(time.time() - start))
            elif (self.dcf_frame_num-1) % self.dcf_params.train_skipping == 0:
                start = time.time()
                self.dcf_filter_optimizer.run(self.dcf_params.CG_iter)
                #print("periodic dcf updating time: {}".format(time.time() - start))

        # Return new state
        # NOTE: if you use a tensor and [[1:0]], which got worse performance, don't know why
        state = [self.target_pos[1] - self.target_sz[1] / 2, self.target_pos[0] - self.target_sz[0] / 2,
                     self.target_pos[1] + self.target_sz[1] / 2, self.target_pos[0] + self.target_sz[0] / 2]

        out['bbox'] =  state

        if out['resized_dcf_heatmap'] is not None:
            dcf_heatmap = out['resized_dcf_heatmap']
            heatmap = cv2.resize(dcf_heatmap, search_image.shape[1::-1])
            heatmap_h = np.clip(heatmap * (30 - 127) * 2.0 + 127, 0, 127)
            heatmap_sv = np.full(search_image.shape[1::-1], 255, dtype=np.uint8)
            heatmap_hsv = np.stack([heatmap_h.astype(np.uint8), heatmap_sv, heatmap_sv], -1)
            out['dcf_heatmap'] = cv2.cvtColor(heatmap_hsv, cv2.COLOR_HSV2BGR)

        return out

    def combine(self, img_shape, prev_pos, prev_sz, scale_z, trtr_outputs, search_image, dcf_flag, dcf_heatmap = None, test_x = None):

        if dcf_heatmap is not None:
            trtr_dcf_scale = 1.0
            resized_bbox = torch.cat([(1 - trtr_dcf_scale) * self.dcf_img_sample_sz / 2, (1 + trtr_dcf_scale) * self.dcf_img_sample_sz / 2])
            # TODO, should not use crop_hwc, please use FFT in fourier.sum_fs
            resized_dcf_heatmap =  crop_hwc(dcf_heatmap.permute(1,2,0).detach().cpu().numpy(), resized_bbox, self.heatmap_size)
            resized_dcf_heatmap /= len(self.dcf_layers)
            unroll_resized_dcf_heatmap = torch.tensor(resized_dcf_heatmap).view(self.heatmap_size * self.heatmap_size)
            best_idx = torch.argmax(unroll_resized_dcf_heatmap)
            # print("the peak {} in dcf heatmap: {}".format(torch.max(unroll_resized_dcf_heatmap), [best_idx % self.heatmap_size, best_idx // self.heatmap_size]))
        else:
            resized_dcf_heatmap = None

        heatmap = trtr_outputs['pred_heatmap'][0].cpu() # we only address with a single image
        raw_heatmap = torch.clone(heatmap).view(self.heatmap_size, self.heatmap_size) # as a image format, for visualize
        found = torch.max(heatmap) > self.score_threshold


        if dcf_heatmap is not None:
            found = True

        if not found:
            return {}

        def change(r):
            return torch.max(r, 1. / r)


        # TODO: 255 is a fixed value because of the training process (template: 127, search: 255)
        bbox_wh_map = trtr_outputs['pred_bbox_wh'][0].cpu() * 255 # convert from relative [0, 1] to absolute [0, height] coordinates

        # scale penalty
        pad = (bbox_wh_map[:, 0] + bbox_wh_map[:, 1]) * get_context_amount()
        sz = torch.sqrt((bbox_wh_map[:, 0] + pad) * (bbox_wh_map[:, 1] + pad))
        s_c = change(sz / get_exemplar_size())

        # aspect ratio penalty
        r_c = change((bbox_wh_map[:, 0] / bbox_wh_map[:, 1]) / (prev_sz[0] / prev_sz[1]) )
        penalty = torch.exp(-(r_c * s_c - 1) * self.size_penalty_k)

        best_idx = 0
        window_factor = self.window_factor
        post_heatmap = None
        best_score = 0

        if dcf_flag == 'hard_negative' and self.hard_negative_recovery:
            window_factor = 0

        # mask a false-positive based on DCF
        if dcf_heatmap is not None and self.init_dcf_score is not None:
            for i in range(self.max_false_postive):
                idx = torch.argmax(heatmap)

                cx = idx.item() % self.heatmap_size
                cy = idx.item() // self.heatmap_size
                lx = max(int(cx - round(self.heatmap_size * 0.05)), 0)
                rx = min(int(cx + round(self.heatmap_size * 0.05)), self.heatmap_size-1)
                ty = max(int(cy - round(self.heatmap_size * 0.05)), 0)
                by = min(int(cy + round(self.heatmap_size * 0.05)), self.heatmap_size-1)

                trtr_score = heatmap[idx].item()
                dcf_score = torch.max(unroll_resized_dcf_heatmap.view(self.heatmap_size, self.heatmap_size)[ty:by, lx:rx]).item()

                if dcf_score / self.init_dcf_score < self.relative_valid_score_threshold * trtr_score / self.init_trtr_score  and dcf_score < self.absolute_valid_score_threshold * torch.max(unroll_resized_dcf_heatmap):
                    #print('false-positive in ({}, {}), trtr score: {} / {}, dcf score: {} / {}, max dcf score: {}'.format(cx, cy, trtr_score, self.init_trtr_score,  dcf_score, self.init_dcf_score, torch.max(unroll_resized_dcf_heatmap)))

                    # mask the score around false positive center
                    heatmap.view(self.heatmap_size, self.heatmap_size)[ty:by, lx:rx] = 0
                else:
                    #print('target in ({}, {}), trtr score: {} / {}, dcf score: {} / {}'.format(cx, cy, trtr_score, self.init_trtr_score,  dcf_score, self.init_dcf_score))

                    # stop search false-positive
                    break

        for i in range(self.window_steps):
            # add distance penalty
            post_heatmap = penalty * heatmap * (1 -  window_factor) + self.window * window_factor
            best_idx = torch.argmax(post_heatmap)
            best_score = heatmap[best_idx].item()


            if best_score > self.score_threshold:
                break;
            else:
                window_factor = np.max(window_factor - self.window_factor / self.window_steps, 0)

            if window_factor == 0:
                post_heatmap = penalty * heatmap

        # print("trtr best score: ", best_score, "; dcf best score: ", torch.max(unroll_resized_dcf_heatmap))

        trtr_heatmap = post_heatmap
        if dcf_heatmap is not None:
            post_heatmap = post_heatmap * (1 -  self.dcf_rate) + unroll_resized_dcf_heatmap * self.dcf_rate
            best_idx = torch.argmax(post_heatmap)
            best_score = post_heatmap[best_idx].item()

        trtr_score = trtr_heatmap[best_idx].item()

        post_heatmap = post_heatmap.view(self.heatmap_size, self.heatmap_size) # as a image format

        # bbox
        ct_int = torch.stack([best_idx % self.heatmap_size, best_idx // self.heatmap_size], dim = -1)
        bbox_reg = trtr_outputs['pred_bbox_reg'][0][best_idx].cpu()
        bbox_ct = (ct_int + bbox_reg)  / float(self.heatmap_size) * self.search_size
        bbox_wh = bbox_wh_map[best_idx]

        ct_delta = (bbox_ct - self.search_size / 2) / scale_z
        cx = prev_pos[0] + ct_delta[0].item()
        cy = prev_pos[1] + ct_delta[1].item()

        # smooth bbox
        lpf = min(best_score * self.size_lpf, 1)
        if self.recovery_flag:
            lpf = 0.5 # heuristic

        bbox_wh = bbox_wh / scale_z
        width = prev_sz[0] * (1 - lpf) + bbox_wh[0].item() * lpf
        height = prev_sz[1] * (1 - lpf) + bbox_wh[1].item() * lpf

        # clip boundary
        bbox = [cx - width / 2, cy - height / 2,
                cx + width / 2, cy + height / 2]

        if self.recovery_flag:
            max_dcf_score = torch.max(dcf_heatmap)
            max_trtr_score = torch.max(heatmap)

            if (max_trtr_score > self.init_trtr_score * self.recovery_score_low_threshold and max_dcf_score > self.init_dcf_score * self.recovery_score_low_threshold ) or max_trtr_score > self.init_trtr_score * self.recovery_score_high_threshold or max_dcf_score > self.init_dcf_score * self.recovery_score_high_threshold:
                self.recovery_flag = False
                print("recovery!!  max score of trtr heatmap: {} / {}, max score of dcf heatmap: {} / {}".format(max_trtr_score, self.init_trtr_score, max_dcf_score, self.init_dcf_score))
            else:
                self.target_sz = self.init_target_sz
                self.target_pos = torch.Tensor([img_shape[0]/2, img_shape[1]/2]) # center of image
                #print("not recovery!!  max score of trtr heatmap: {} / {}, max score of dcf heatmap: {} / {}".format(max_trtr_score, self.init_trtr_score, max_dcf_score, self.init_dcf_score))

                return {
                    'bbox': bbox,
                    'score': best_score,
                    'search_image': search_image,
                    'resized_dcf_heatmap': resized_dcf_heatmap,
                }


        # check boundary issue
        margin = np.array(img_shape) * self.boundary_margin
        if bbox[0] <= margin[1] or bbox[1] < margin[0] or bbox[2] > img_shape[1]-1 - margin[1] or bbox[3] > img_shape[0]-1 - margin[0]:
            if self.boundary_recovery:
                self.invalid_bbox_cnt += 1

                # too small bbox, instant recovery
                if bbox[0] <= margin[1] or bbox[2] > img_shape[1]-1 - margin[1]:
                    if width <= self.hard_size_margin * img_shape[1]:
                        self.recovery_flag = True
                if bbox[1] <= margin[0] or bbox[3] > img_shape[0]-1 - margin[0]:
                    if height <= self.hard_size_margin * img_shape[0]:
                        self.recovery_flag = True
                if width <= self.relax_size_margin * img_shape[1] and height <= self.relax_size_margin * img_shape[0]:
                    self.recovery_flag = True
            else:
                # too small bbox, instant recovery
                if width <= self.hard_size_margin * img_shape[1] and height <= self.hard_size_margin * img_shape[0]:
                    self.recovery_flag = True

            if self.invalid_bbox_cnt > self.invalid_bbox_cnt_max and not self.recovery_flag:

                dcf_heatmap = self.dcf_localize_target(operation.conv2d(test_x, self.init_dcf_filter, mode='same'))[1][0]
                dcf_heatmap /= len(self.dcf_layers)
                check_region = torch.tensor([self.search_size //2 - int(height * scale_z / 2),
                                             self.search_size //2 - int(width * scale_z / 2),
                                             self.search_size //2 + int(height * scale_z / 2),
                                             self.search_size //2 + int(width * scale_z / 2)])

                dcf_score = torch.max(dcf_heatmap[0][check_region[1]:check_region[3],check_region[0]:check_region[2]])

                check_region = check_region.float() * self.heatmap_size / self.search_size
                check_region = check_region.int()
                trtr_score = torch.max(heatmap.view(self.heatmap_size, self.heatmap_size)[check_region[1]:check_region[3],check_region[0]:check_region[2]])

                #print("boundary!!! max score of trtr heatmap: {} / {}, max score of dcf heatmap: {} / {}, count: {}, margin: {}, size: {}".format(trtr_score, self.init_trtr_score, dcf_score, self.init_dcf_score, self.invalid_bbox_cnt, margin, [width, height]))

                #if trtr_score < 0.1:
                if dcf_score < self.boundary_target_score_threshold * self.init_dcf_score and trtr_score < self.boundary_target_score_threshold * self.init_trtr_score:
                    self.invalid_bbox_score_cnt += 1
                else:
                    # reset
                    self.invalid_bbox_score_cnt = 0

                if self.invalid_bbox_score_cnt > self.invalid_bbox_score_cnt_max:
                    self.recovery_flag = True

            if self.recovery_flag:
                # reset the dcf optimizer
                self.dcf_filter_optimizer.residuals = self.init_dcf_filter_optimizer_residuals
                self.dcf_filter_optimizer.losses = self.init_dcf_filter_optimizer_losses
                self.dcf_filter = copy.deepcopy(self.init_dcf_filter)

                self.dcf_num_stored_samples = self.dcf_num_init_samples.copy()
                self.dcf_previous_replace_ind = [None] * len(self.dcf_num_stored_samples)
                for train_samp, y_memory, sw, init_sw, num in zip(self.dcf_training_samples, self.dcf_y, self.dcf_sample_weights, self.dcf_init_sample_weights, self.dcf_num_init_samples):
                    train_samp[num:] = 0
                    y_memory[num:] = 0
                    sw[:num] = init_sw
                    sw[num:] = 0


                self.target_sz = self.init_target_sz
                self.target_pos = torch.Tensor([img_shape[0]/2, img_shape[1]/2]) # center of image

                print("reset the tracker becuase of bbox near boundary. max_heatmap score from trtr: {}, count: {}".format(torch.max(heatmap), self.invalid_bbox_cnt))

                self.invalid_bbox_cnt = 0
                self.invalid_bbox_score_cnt = 0

                return {
                    'bbox': bbox,
                    'score': best_score,
                    'search_image': search_image,
                    'resized_dcf_heatmap': resized_dcf_heatmap,
                }
        else:
            self.invalid_bbox_cnt = 0

        bbox = self._bbox_clip(bbox, img_shape)

        # udpate state ([y,x])
        self.target_pos = torch.Tensor([(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2])
        self.target_sz = torch.Tensor([bbox[3] - bbox[1], bbox[2] - bbox[0]])

        # debug for search image:
        debug_bbox = torch.round(box_cxcywh_to_xyxy(torch.cat([bbox_ct, bbox_wh * scale_z]))).int()
        rec_search_image = cv2.rectangle(search_image,
                                         (debug_bbox[0], debug_bbox[1]),
                                         (debug_bbox[2], debug_bbox[3]),(0,255,0),3)

        raw_heatmap = (torch.round(raw_heatmap * 255)).detach().numpy().astype(np.uint8)
        post_heatmap = (torch.round(post_heatmap * 255)).detach().numpy().astype(np.uint8)
        heatmap_resize = cv2.resize(raw_heatmap, search_image.shape[1::-1])

        heatmap_h = np.clip(heatmap_resize / -255 * (127 - 30) * 2  + 127, 0, 127)
        heatmap_sv = np.full(search_image.shape[1::-1], 255, dtype=np.uint8)
        heatmap_hsv = np.stack([heatmap_h.astype(np.uint8), heatmap_sv, heatmap_sv], -1)
        heatmap_color = cv2.cvtColor(heatmap_hsv, cv2.COLOR_HSV2BGR)
        rec_search_image = np.round(0.4 * heatmap_color + 0.6 * rec_search_image.copy()).astype(np.uint8) 


        ## import to be here
        self.dcf_target_scale = 1 / siamfc_like_scale(bbox)[1]

        if self.init_trtr_score is None:
            self.init_trtr_score = trtr_score
        if self.init_dcf_score is None and dcf_heatmap is not None:
            if unroll_resized_dcf_heatmap[best_idx].item() > 0:
                self.init_dcf_score = unroll_resized_dcf_heatmap[best_idx].item()

        return {
            'bbox': bbox,
            'score': best_score,
            'trtr_raw_heatmap': heatmap_color, #raw_heatmap,
            'trtr_post_heatmap': post_heatmap,
            'search_image': search_image, # debug
            'resized_dcf_heatmap': resized_dcf_heatmap,
            'trtr_score': trtr_score,
            'bbox_in_search_image': bbox_ct
        }


    def dcf_apply_filter(self, sample_x: TensorList):
        #print("len of dcf_filter: ", len(self.dcf_filter))
        return operation.conv2d(sample_x, self.dcf_filter, mode='same')

    def dcf_localize_target(self, scores_raw):
        # TODO: learn the mechanism
        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) * scores_raw.size(3))
        #print("scores_raw", type(scores_raw), len(scores_raw), scores_raw[0].shape)

        for i, sz in enumerate(self.dcf_feature_sz_list):
            sf_weighted[i] = fourier.shift_fs(sf_weighted[i], math.pi * (1 - torch.Tensor([self.dcf_kernel_size[0]%2, self.dcf_kernel_size[1]%2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.dcf_output_sz)

        scores *= self.dcf_output_window

        #print("scores", type(scores), scores.shape)

        sz = scores.shape[-2:]

        # Shift scores back
        scores = torch.cat([scores[...,(sz[0]+1)//2:,:], scores[...,:(sz[0]+1)//2,:]], -2)
        scores = torch.cat([scores[...,:,(sz[1]+1)//2:], scores[...,:,:(sz[1]+1)//2]], -1)

        # Get the average heatmap
        # scores /= len(self.dcf_layers) // should be here, but, we have a special resize process in self.combine: crop_hwc (actually, this function is not good)

        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        #print(max_score1.shape, max_disp1.shape, max_score1, max_disp1)

        max_score1 = max_score1[0]
        max_disp1 = max_disp1[0,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - self.dcf_output_sz // 2
        translation_vec1 = target_disp1 * (self.dcf_img_support_sz / self.dcf_output_sz) * self.dcf_target_scale

        if max_score1.item() < self.dcf_params.target_not_found_threshold:
            return translation_vec1, scores, 'not_found'

        # Mask out target neighborhood
        target_neigh_sz = self.dcf_params.target_neighborhood_scale * self.target_sz / self.dcf_target_scale
        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores[0:1,...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.dcf_output_sz // 2
        translation_vec2 = target_disp2 * (self.dcf_img_support_sz / self.dcf_output_sz) * self.dcf_target_scale

        # Handle the different cases
        if max_score2 > self.dcf_params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1**2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2**2))
            disp_threshold = self.dcf_params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scores, 'uncertain'

        if max_score2 > self.dcf_params.hard_negative_threshold * max_score1 and max_score2 > self.dcf_params.target_not_found_threshold:
            return translation_vec1, scores, 'hard_negative'

        return translation_vec1, scores, None


    def dcf_project_sample(self, x: TensorList, proj_matrix = None):
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.dcf_projection_matrix
        return operation.conv2d(x, proj_matrix).apply(self.dcf_projection_activation)

    def dcf_init_learning(self):

        # Filter regularization
        self.dcf_filter_reg = self.dcf_fparams.attribute('filter_reg')

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = self.dcf_params.get('projection_activation', 'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.dcf_projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.dcf_projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.dcf_projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.dcf_projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = self.dcf_params.get('response_activation', 'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.dcf_response_activation = lambda x: x
        elif response_activation == 'relu':
            self.dcf_response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.dcf_response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.dcf_response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')


    def dcf_generate_init_samples(self, im: np.ndarray, target_pos, target_scale, padding_value: float):
        """Generate augmented initial samples."""

        # Compute augmentation size
        aug_expansion_factor = self.dcf_params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.dcf_img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.dcf_img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.dcf_img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.dcf_img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = self.dcf_params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.dcf_img_sample_sz * random_shift_factor).long().tolist()

        # Create transformations
        self.dcf_transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.dcf_params.augmentation:
            self.dcf_transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.dcf_params.augmentation['shift']])
        if 'relativeshift' in self.dcf_params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.dcf_img_sample_sz/2).long().tolist()
            self.dcf_transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.dcf_params.augmentation['relativeshift']])
        if 'fliplr' in self.dcf_params.augmentation and self.dcf_params.augmentation['fliplr']:
            self.dcf_transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.dcf_params.augmentation:
            self.dcf_transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.dcf_params.augmentation['blur']])
        if 'scale' in self.dcf_params.augmentation:
            self.dcf_transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.dcf_params.augmentation['scale']])
        if 'rotate' in self.dcf_params.augmentation:
            self.dcf_transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.dcf_params.augmentation['rotate']])

        # Generate initial samples
        init_samples = self.dcf_extract_transformed(im, target_pos, target_scale, aug_expansion_sz, self.dcf_transforms, padding_value)

        # Add dropout samples
        if 'dropout' in self.dcf_params.augmentation:
            num, prob = self.dcf_params.augmentation['dropout']
            self.dcf_transforms.extend(self.dcf_transforms[:1]*num)
            for i in range(len(init_samples)):
                init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        #print("layer1 init sample: ", init_samples[0].shape) # debug

        return TensorList(init_samples)

    def dcf_init_label_function(self, train_x, target_pos, target_sz, target_scale):
        # Allocate label function
        self.dcf_y = TensorList([x.new_zeros(self.dcf_params.sample_memory_size, 1, x.shape[2], x.shape[3]) for x in train_x])

        # Output sigma factor
        output_sigma_factor = self.dcf_fparams.attribute('output_sigma_factor')
        base_target_sz = target_sz / target_scale
        self.dcf_sigma = (self.dcf_feature_sz_list / self.dcf_img_support_sz * base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized coords (offset becuase of the float)
        target_center_norm = (target_pos - target_pos.round()) / (target_scale * self.dcf_img_support_sz)

        # Generate label functions
        for y, sig, sz, x in zip(self.dcf_y, self.dcf_sigma, self.dcf_feature_sz_list, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(self.dcf_kernel_size[0] + 1) % 2, (self.dcf_kernel_size[1] + 1) % 2])
            for i, T in enumerate(self.dcf_transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.dcf_img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)

        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.dcf_y, train_x)])

    def dcf_init_memory(self, train_x):
        # Initialize first-frame training samples
        self.dcf_num_init_samples = train_x.size(0)
        #print("self.dcf_num_init_samples: ", self.dcf_num_init_samples)
        self.dcf_init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.dcf_init_training_samples = train_x

        # Sample counters and weights
        self.dcf_num_stored_samples = self.dcf_num_init_samples.copy()
        self.dcf_previous_replace_ind = [None] * len(self.dcf_num_stored_samples) #TODO: need for layers?
        #print("self.dcf_previous_replace_ind: ", self.dcf_previous_replace_ind)
        self.dcf_sample_weights = TensorList([x.new_zeros(self.dcf_params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.dcf_sample_weights, self.dcf_init_sample_weights, self.dcf_num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.dcf_training_samples = TensorList(
            [x.new_zeros(self.dcf_params.sample_memory_size, self.dcf_compressed_dim, x.shape[2], x.shape[3]) for x in train_x])

    def dcf_update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate = None):
        replace_ind = self.dcf_update_sample_weights(self.dcf_sample_weights, self.dcf_previous_replace_ind, self.dcf_num_stored_samples, self.dcf_num_init_samples, self.dcf_fparams, learning_rate)
        self.dcf_previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.dcf_training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x
        for y_memory, y, ind in zip(self.dcf_y, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y
        self.dcf_num_stored_samples += 1


    def dcf_update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams, learning_rate = None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def dcf_get_label_function(self, target_pos, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (target_pos - sample_pos) / (sample_scale * self.dcf_img_support_sz)
        for sig, sz in zip(self.dcf_sigma, self.dcf_feature_sz_list):
            center = sz * target_center_norm + 0.5 * torch.Tensor([(self.dcf_kernel_size[0] + 1) % 2, (self.dcf_kernel_size[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

    def dcf_extract_transformed(self, image, pos, scale, image_sz, transforms, padding_value):
        """Extract features from a set of transformed image samples.
        args:
            im: Image.
            pos: Center position for extraction.
            scale: Image scale to extract features from.
            image_sz: Size to resize the image samples to before extraction.
            transforms: A set of image transforms to apply.
            padding_value: the padding value for outside are in color 3 channel
        """

        # Apply transforms
        assert image_sz[0] == image_sz[1] # TODO: change utils.crop_hwc
        sz = scale * image_sz[0]
        bbox  = [pos[1]-sz/2, pos[0]-sz/2, pos[1]+sz/2, pos[0]+sz/2]
        crop_image = crop_hwc(image, bbox, int(image_sz[0]), padding = padding_value)

        im_patch =  torch.from_numpy(crop_image).float().permute(2, 0, 1).unsqueeze(0) # [1, 3, h, w]
        im_patches = torch.cat([T(im_patch) for T in transforms])

        # Compute features
        im_patches = im_patches / 255
        im_patches -= torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        im_patches /= torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        im_patches = im_patches.cuda()

        mask = [0, 0, im_patches.shape[-2], im_patches.shape[-1]]
        with torch.no_grad():
            features = self.model.backbone(nested_tensor_from_tensor_list(im_patches, [torch.as_tensor(mask).float()] * im_patches.shape[0]))[2]

        return self.dcf_feature_preprocess(features)

    @torch.no_grad()
    def dcf_feature_preprocess(self, features):
        """Get the pre-processed featuure for DCF .
        args:
            feature: output from the backbone [x, c, w, h]
        """

        dcf_features = []
        for layer in self.dcf_layers:
            feature = features[layer[-1]]

            if self.model.backbone.dilation:  # i.e., Resnet50
                feature = F.adaptive_avg_pool2d(feature, self.dcf_feature_sz)

            # Normalize
            normalize_power = 2 # heuristic
            feature /= (torch.sum(feature.abs().view(feature.shape[0],1,1,-1)**normalize_power, dim=3, keepdim=True) /
                                   (feature.shape[1]*feature.shape[2]*feature.shape[3]) + 1e-10)**(1/normalize_power)
            dcf_features.append(feature)

        return TensorList(dcf_features)

def get_args_parser():
    # Inherit from Baseline Tracker
    parser = baseline_tracker_args_parser()

    # * hyper-parameter for tracking
    dcf_parser = ArgumentParser(prog='online dcf')
    dcf_parser.add_argument('--layers', default=[], nargs='+',
                        help = 'layers of backbone used by dcf')
    dcf_parser.add_argument('--rate', type=float, default=0.6,
                        help='the weight for integrate dcf and trtr for heatmap ')
    dcf_parser.add_argument('--size', type=int, default=0,
                        help='the size for feature for dcf')
    dcf_parser.add_argument('--sample_memory_size', type=int, default=250,
                        help='the size of the trainining sample for DCF ')
    parser.add_argument('--dcf', action=ActionParser(parser=dcf_parser))


    # Post Process
    parser.add_argument('--postprocess.boundary_recovery', type=bool, default=False,
                                    help='whether use boundary recovery')
    parser.add_argument('--postprocess.hard_negative_recovery', type=bool, default=False,
                                    help='(Depracated) whether use hard negative recovery')
    parser.add_argument('--postprocess.lost_target_recovery', type=bool, default=False,
                                    help='whether use lost target recovery')

    ## Lost Traget
    lost_target_parser = ArgumentParser(prog='lost target')
    lost_target_parser.add_argument('--boundary_margin', type=float, default=0.3,
                                    help='the margin to the boundary of image')
    lost_target_parser.add_argument('--translation_threshold', type=float, default=0.03,
                                    help='the translation threshold to determine static tracking')
    lost_target_parser.add_argument('--cnt_threshold', type=int, default=60,
                                    help='the count threshold to determine static tracking')
    lost_target_parser.add_argument('--score_threshold', type=float, default=0.5,
                                    help='the score threshold to determine static tracking')
    parser.add_argument('--postprocess.lost_target', action=ActionParser(parser=lost_target_parser))


    return parser

def build_tracker(args):

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in Pytorch")

    device = torch.device('cuda')

    model, _, postprocessors = build_model(args.model)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    assert 'model' in checkpoint
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    if len(args.dcf.layers) == 0:
        args.dcf.layers = ['layer2', 'layer3']

    return Tracker(model,
                   postprocessors["bbox"],
                   args.search_size,
                   args.dcf,
                   args.postprocess)
