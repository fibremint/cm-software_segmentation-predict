

import os

import numpy as np
import scipy.misc as misc
import openslide


class SlideLoader():
    def __init__(self, batch_size, to_real_scale=4, level=1, imsize=512):

        # assert level ==0, 'level other than 0 is not supported'
        # The acutally size for each sample is imsize / scale, 
        # then resized to imsize for network input
        self.imsize = imsize
        self.batch_size = batch_size
        self.level = level
        self.scale = 1.0 / to_real_scale
        
    def iterator(self, slide, loc):
        idx = 0
        for i in range(0, len(loc)):
            imsize = int((self.actual_crop_size) * self.scale)
            batch_img = np.zeros((self.batch_size, imsize, imsize, 3), dtype=np.float16)
            cur_left = len(loc) - idx
            if cur_left == 0: break
            act_batch_size = min(cur_left, self.batch_size)
            all_loc = [] # loc after scaled down. It is used to stich the patches back
            for b in range(act_batch_size):
                y, x = loc[idx] 
                all_loc.append((int(y*self.scale),int(x*self.scale)))
                origin_img = slide.read_region((x,y), self.level, (self.actual_crop_size, self.actual_crop_size))
                origin_img = np.asarray(origin_img)[:,:,0:-1] # remove the alpha channel
                origin_img = misc.imresize(origin_img, self.scale, interp='nearest')

                batch_img[b] = origin_img
                idx += 1
            yield batch_img[:act_batch_size], all_loc
                    
    def get_slide_iterator(self, path, down_scale_rate, overlapp=128):
        # start_pos: (y, x)
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]

        slide_pointer = openslide.open_slide(path)
        w, h = slide_pointer.level_dimensions[self.level]
        # recalculate
        if down_scale_rate > 1:
            level = 0
        else:
            level = self.level
        self.level_ratio = slide_pointer.level_dimensions[0][0] // slide_pointer.level_dimensions[level][0]
        self.scale = self.scale * self.level_ratio
        self.actual_crop_size = int(self.imsize / self.scale)

        n_width = w // (self.actual_crop_size - overlapp)
        n_height = h // (self.actual_crop_size - overlapp)

        out_w = w * self.scale  # int(n_width * self.actual_crop_size * self.scale)
        out_h = h * self.scale  # int(n_height * self.actual_crop_size * self.scale)

        # compute location index
        loc = []

        for i in range(n_height + 2):
            for j in range(n_width + 2):
                y = i * (self.actual_crop_size - overlapp)
                x = j * (self.actual_crop_size - overlapp)
                y = max(y, 0)
                x = max(x, 0)
                if y + self.actual_crop_size > h:
                    y = h - self.actual_crop_size
                if x + self.actual_crop_size > w:
                    x = w - self.actual_crop_size

                loc.append((y, x))
        print('Iterator: ({}, {}) at level {} (ratio {}, actual_size {} (overlap {}))'.format(h, w, self.level,
                                                                                              self.level_ratio,
                                                                                              self.actual_crop_size,
                                                                                              overlapp))
        num_batches = len(loc) // self.batch_size

        print('{} patches, splited to {} batches'.format(len(loc), num_batches))

        return self.iterator(slide_pointer, loc), num_batches, name, [int(out_h), int(out_w)], len(loc)
