# coding=utf-8
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from utils.windows import slide_window, search_windows
from utils.boxes import draw_labeled_boxes, add_heat, apply_threshold

PERCENT = 5


def process_video(in_file, out_file, windows_cfg, svc, X_scaler, color_space, spatial_size,
    hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):
    frame_count = 0
    last_labels = None

    def process_img(img):
        nonlocal frame_count, last_labels, windows_cfg, svc, X_scaler, color_space, spatial_size, \
            hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat

        res = np.copy(img)
        img = img.astype(np.float32)/255

        frame_count += 1

        if frame_count == 1 or frame_count % PERCENT == 0:
            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            all_windows = []

            for wc in windows_cfg:
                windows = slide_window(img, x_start_stop = wc.get('x_start_stop'),
                    y_start_stop = wc.get('y_start_stop'), xy_window = wc.get('xy_window'), xy_overlap = wc.get('xy_overlap'))
                hot_windows = search_windows(img, windows, svc, X_scaler, color_space = color_space, spatial_size = spatial_size,
                    hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,
                    hog_channel = hog_channel, spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)

                for w in hot_windows:
                    all_windows.append(w)

            heat = add_heat(heat, all_windows)
            heat = apply_threshold(heat, 1)
            labels = label(heat)
            last_labels = labels

        return draw_labeled_boxes(res, last_labels)

    clip = VideoFileClip(in_file)
    clip = clip.fl_image(process_img)
    clip.write_videofile(out_file, audio=False)


