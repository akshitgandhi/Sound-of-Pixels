import os
import random
from .base import BaseDataset
import numpy as np

labels = {
        'accordion' : 0,
        'acoustic_guitar' : 1,
        'cello' : 2,
        'clarinet' : 3,
        'erhu' : 4,
        'flute' : 5,
        'saxophone' : 6,
        'trumpet' : 7,
        'tuba' : 8,
        'violin' : 9,
        'xylophone' : 10
        }


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.instruments = labels

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        gt_labels = [None for n in range(N)] #Quan

        # the first video
        infos[0] = self.list_sample[index]
        instrument_0 = ""
        for instrument in self.instruments.keys():
            if instrument in infos[0][0]:
                instrument_0 = instrument
                break

        assert(instrument_0 is not "")

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            while True:
                indexN = random.randint(0, len(self.list_sample)-1)
                if instrument_0 not in self.list_sample[indexN][0]:
                    infos[n] = self.list_sample[indexN]
                    # print("instrument_0 = {}, path_audio[1] = {}".format(instrument_0, infos[n][0]))
                    break


        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

            gt_labels[n] = -1
            for instrument in self.instruments.keys():
                if instrument in path_audioN:
                    gt_labels[n] = self.instruments[instrument]
                    break
                    
            if gt_labels[n] == -1:
                print("path_audioN = {}".format(path_audioN))
                raise Exception("invalid instrument")


        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):                             
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps                            
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)
        # except ValueError as e:
        #     print('ValueError while loading audio or frame at path {}, last center timeN = {}: {}'.format(last_audio_path, last_center_timeN, e))
        #     mag_mix, mags, frames, audios, phase_mix = \
        #         self.dummy_mix_data(N)
        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        ret_dict['gt_labels'] = gt_labels #Quan testing - to be removed when testing is done!!!
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
            ret_dict['path_frames'] = path_frames
            ret_dict['gt_labels'] = gt_labels

        return ret_dict
