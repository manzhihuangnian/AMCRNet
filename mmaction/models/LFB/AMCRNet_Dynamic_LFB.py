from collections import defaultdict
from mmaction.models.builder import LFB
import numpy as np

@LFB.register_module()
class Dynamic_Feature_Bank(object):
    def __init__(self,
                 window_size=11,
                 max_person_mun_persec=5
                 ):
        self.window_size=window_size
        self.cache = defaultdict(dict)
        self.max_person_mun_persec=max_person_mun_persec

    def update(self, update_info):
        for movie_id, feature_per_movie in update_info.items():
            self.cache[movie_id].update(feature_per_movie)

    def update_list(self, update_info_list):
        for update_info in update_info_list:
            self.update(update_info)

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item)==2:
            return self.cache[item[0]][item[1]]
        return self.cache[item]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key)==2:
            self.cache[key[0]][key[1]] = value
        else:
            self.cache[key] = value

    def __delitem__(self, item):
        if isinstance(item, tuple) and len(item)==2:
            del self.cache[item[0]][item[1]]
        else:
            del self.cache[item]

    def __contains__(self, item):
        if isinstance(item, tuple) and len(item)==2:
            return (item[0] in self.cache and item[1] in self.cache[item[0]])
        return (item in self.cache)

    def items(self):
        return self.cache.items()

    def get_memory_feature_twobranch(self, img_metas, forward_loss_list):
        lt_high_features_list = []
        lt_roi_features_list = []
        lt_index_list = []
        for index, img_meta in enumerate(img_metas):
            cur_video_high_features=[]
            cur_video_roi_features=[]
            cur_video_index_list=[]
            video_id, timestamp= img_meta["img_key"].split(",")
            timestamp=int(timestamp)
            video_features = self.cache[video_id]
            window_size, K = self.window_size, self.max_person_mun_persec
            start = timestamp - (window_size // 2)
            forward_loss=forward_loss_list[index]

            for idx, sec in enumerate(range(start, start + window_size)):
                if sec in video_features and sec != timestamp:
                    sample_high_features = []
                    sample_roi_features=[]
                    single_clip_high_features, loss_tag = video_features[sec]["high_relation"], \
                                                       video_features[sec]["loss_tag"]
                    num_feat = len(single_clip_high_features)
                    factor = min(loss_tag.item() / forward_loss.item(), forward_loss.item() / loss_tag.item())
                    single_clip_high_features = [feat * factor for feat in single_clip_high_features]

                    single_clip_roi_features = video_features[sec]["update_feature"]
                    single_clip_roi_features = [feat * factor for feat in single_clip_roi_features]

                    if self.max_person_mun_persec > 2 and self.max_person_mun_persec < num_feat:
                        random_lfb_indices = np.random.choice(
                            range(num_feat), self.max_person_mun_persec, replace=False)
                        for indices in random_lfb_indices:
                            sample_high_features.append(single_clip_high_features[indices])
                            sample_roi_features.append(single_clip_roi_features[indices])
                    else:
                        sample_high_features = single_clip_high_features
                        sample_roi_features = single_clip_roi_features

                    sample_high_features=[feat.cuda() for feat in sample_high_features]
                    sample_roi_features=[feat.cuda() for feat in sample_roi_features]

                    cur_video_high_features.append(sample_high_features)
                    cur_video_roi_features.append(sample_roi_features)
                    cur_video_index_list.append(idx)

            lt_high_features_list.append(cur_video_high_features)
            lt_roi_features_list.append(cur_video_roi_features)
            lt_index_list.append(cur_video_index_list)

        return lt_high_features_list, lt_roi_features_list, lt_index_list

    def get_memory_feature_onebranch(self, img_metas, forward_loss_list):
        lt_high_features_list = []
        lt_index_list = []
        for index,img_meta in enumerate(img_metas):
            cur_video_high_features = []
            cur_video_index_list = []
            video_id, timestamp = img_meta["img_key"].split(",")
            timestamp=int(timestamp)
            video_features = self.cache[video_id]
            window_size, K = self.window_size, self.max_person_mun_persec
            start = timestamp - (window_size // 2)
            forward_loss=forward_loss_list[index]

            for idx, sec in enumerate(range(start, start + window_size)):
                if sec in video_features and sec != timestamp:
                    sample_high_feature = []
                    single_clip_high_features, loss_tag = video_features[sec]["high_relation"], \
                                                          video_features[sec]["loss_tag"]

                    num_feat = len(single_clip_high_features)
                    factor = min(loss_tag.item() / forward_loss.item(), forward_loss.item() / loss_tag.item())
                    single_clip_high_features = [feat * factor for feat in single_clip_high_features]

                    if self.max_person_mun_persec > 2 and self.max_person_mun_persec < num_feat:
                        random_lfb_indices = np.random.choice(
                            range(num_feat), self.max_person_mun_persec, replace=False)
                        for indices in random_lfb_indices:
                            sample_high_feature.append(single_clip_high_features[indices])

                    else:
                        sample_high_feature = single_clip_high_features


                    sample_high_feature=[feat.cuda() for feat in sample_high_feature]
                    cur_video_high_features.append(sample_high_feature)
                    cur_video_index_list.append(idx)
            lt_high_features_list.append(cur_video_high_features)
            lt_index_list.append(cur_video_index_list)

        return lt_high_features_list, lt_index_list



