import numpy as np
import cv2


def overlay_rectangle(img, rect, color=(0, 255, 0), line_width=2):
    if rect is not None:
        tl_ = (int(round(rect[0])), int(round(rect[1])))
        br_ = (int(round(rect[0] + rect[2])), int(round(rect[1] + rect[3])))
        cv2.rectangle(img, tl_, br_, color, line_width)

def overlay_progress(img, frame_idx, sequence_len, bg_color=(120, 120, 120), bar_color=(255, 255, 255), bar_width=5):
    img_h, img_w, _ = img.shape
    # draw background bar
    tl_ = (0, img_h - bar_width - 1)
    br_ = (img_w - 1, img_h - 1)
    cv2.rectangle(img, tl_, br_, bg_color, bar_width)
    # draw progress bar
    x_ = (frame_idx / sequence_len) * (img_w - 1)
    br_ = (int(round(x_)), img_h - 1)
    cv2.rectangle(img, tl_, br_, bar_color, bar_width)

def overlay_mask(img, mask, color=(0, 255, 0), line_width=2, alpha=0.6):
    if mask is not None:
        m = mask.astype(np.float32)
        m_bin = m > 0.5
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        img_r[m_bin] = alpha * img_r[m_bin] + (1 - alpha) * color[0]
        img_g[m_bin] = alpha * img_g[m_bin] + (1 - alpha) * color[1]
        img_b[m_bin] = alpha * img_b[m_bin] + (1 - alpha) * color[2]

        # draw contour around mask
        M = m_bin.astype(np.uint8)
        if cv2.__version__[0] == '4':
            contours, _ = cv2.findContours(M, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(M, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color, line_width)

class Visualizer():
    def __init__(self, seq_len=None):
        self.window_name = 'Window'
        self.timeout = 0
        self.sequence_length = seq_len
        self.window_initialized = False

        self.color_pallete = [(255, 0, 0), (255, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), 
                              (60, 179, 113), (238, 130, 238), (230, 126, 34), (204, 204, 255)]
    
    def _resize_vis_img(self, img):
        if max(img.shape) > 1000:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        return img

    def visualize(self, img_rgb, mask=None, bbox=None, frame_index=None, distractors=None, gt=None):
        
        if mask is not None:
            alpha_ = 0.6
            line_width_ = 1
            if isinstance(mask, list):
                for i, m in enumerate(mask):
                    color_ = self.color_pallete[i]
                    overlay_mask(img_rgb, m, color=color_, alpha=alpha_, line_width=line_width_)
            else:
                color_ = self.color_pallete[0]
                overlay_mask(img_rgb, mask, color=color_, alpha=alpha_, line_width=line_width_)
        
        if bbox is not None:
            alpha_ = 0.6
            line_width_ = 1
            if isinstance(bbox[0], list):
                for i, bb in enumerate(bbox):
                    color_ = self.color_pallete[i]
                    # overlay_mask(img_rgb, m, color=color_, alpha=alpha_, line_width=line_width_)
                    overlay_rectangle(img_rgb, bb, color=color_, line_width=line_width_)
            else:
                color_ = self.color_pallete[0]
                # overlay_mask(img_rgb, mask, color=color_, alpha=alpha_, line_width=line_width_)
                overlay_rectangle(img_rgb, bbox, color=color_, line_width=line_width_)
        
        # draw groundtruth
        if gt is not None:
            overlay_rectangle(img_rgb, gt, color=(0, 200, 0), line_width=1)

        #draw progress bar
        if self.sequence_length is not None and frame_index is not None:
            overlay_progress(img_rgb, frame_index, self.sequence_length)

        # resize image if it is too large
        img_rgb = self._resize_vis_img(img_rgb)

        img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if not self.window_initialized:
            # cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        cv2.imshow(self.window_name, img_vis)
        key_ = cv2.waitKey(self.timeout)

        if key_ == 27:
            # Esc
            cv2.destroyAllWindows()
            exit(1)
        elif key_ == 32:
            # Space
            if self.timeout == 0:
                self.timeout = 1
            else:
                self.timeout = 0
