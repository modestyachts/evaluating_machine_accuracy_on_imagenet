import dateparser
import collections

import numpy as np


def load_ha(ha_data):
    return HumanAccuracyPortable(annot_data=ha_data["annot_data"], 
                                 labeled_images=ha_data["labeled_images"], 
                                 image_labels=ha_data["image_labels"],
                                 split_wnids=ha_data["split_wnids"],
                                 image_map=ha_data["image_map"],
                                 all_candidates=ha_data["all_candidates"],
                                 wnid_map=ha_data["wnid_map"],
                                 initial_annotations=ha_data["initial_annots"])

class HumanAccuracyPortable(object):
    def __init__(self, annot_data, labeled_images, image_map, image_labels, split_wnids, all_candidates, wnid_map, initial_annotations):
        self._image_map = image_map
        self.labeled_images = labeled_images
        self.image_labels = image_labels
        self.annot_data  = annot_data
        self.all_candidates = all_candidates
        self.initial_annotations = initial_annotations
        self.PROBLEMATIC_FILTER = lambda image, image_labels, user_label: not image["problematic"]
        self.V2_FILTER = lambda image, image_labels, user_label: image["id"] in self.all_candidates
        self.VAL_FILTER = lambda image, image_labels, user_label: not image["id"] in self.all_candidates
        self.NO_FILTER = lambda image, image_labels, user_label: True

        self.split_wnids = split_wnids
        self.wnid_map = wnid_map
        self.rev_wnid_map = collections.defaultdict(list)
        for k,v in self.wnid_map.items():
            self.rev_wnid_map[v].append(k)
        self.fast_imgs = set()
        self.slow_imgs = set()

        def NOTDOG_FILTER(image, image_labels, user_label):
            wnid = self.wnid_map[image['id']]
            if wnid in self.split_wnids[('organism', 'dog')]:
                return False
            else:
                return True
        self.NOTDOG_FILTER = NOTDOG_FILTER

        def FASTIMAGES_FILTER(image, image_labels, user_label):

            if image['id'] in self.fast_imgs:
                return True
            elif image['id'] in self.slow_imgs:
                return False

            human_users = ['human_a', 'human_b', 'human_c', 'human_d', 'human_e']

            times_spent = []
            for user in human_users:
                start_time = dateparser.parse(self.annot_data[user][image['id']]['extra_info']['start_time'])
                end_time = dateparser.parse(self.annot_data[user][image['id']]['extra_info']['end_time'])
                times_spent.append((end_time - start_time).seconds)

            if np.median(times_spent) < 60:
                self.fast_imgs.add(image['id'])
                return True
            else:
                self.slow_imgs.add(image['id'])
                return False

        self.FASTIMAGES_FILTER = FASTIMAGES_FILTER


        def NOT_PROBLEMATIC(image, image_labels, user_label):
            if image['problematic']:
                return False
            else:
                return True
        self.NOT_PROBLEMATIC = NOT_PROBLEMATIC


        def OBJECT_FILTER(image, image_labels, user_label):
            wnid = self.wnid_map[image['id']]
            if wnid in self.split_wnids[('object',)]:
                return True
            else:
                return False
        self.OBJECT_FILTER = OBJECT_FILTER

        def ORGANISM_FILTER(image, image_labels, user_label):
            wnid = self.wnid_map[image['id']]
            if wnid in self.split_wnids[('organism',)]:
                return True
            else:
                return False
        self.ORGANISM_FILTER = ORGANISM_FILTER

        def VAL_SET(image, image_labels, user_label):
            prefix = "ILSVRC2012_val_"
            len_prefix = len(prefix)
            if image['id'][:len_prefix] == prefix:
                return True
            else:
                return False
        self.VAL_SET = VAL_SET

        def VTWO_SET(image, image_labels, user_label):
            return not VAL_SET(image, image_labels, user_label)
        self.VTWO_SET = VTWO_SET



        def ORGANISM_NOT_DOG_FILTER(image, image_labels, user_label):
            wnid = self.wnid_map[image['id']]
            if wnid in self.split_wnids[('object')]:
                return True
            else:
                return False
        self.ORGANISM_NOT_DOG_FILTER = ORGANISM_NOT_DOG_FILTER


    def grade_predictions(self, username, filters=[], filter_join=all, correct_state=["correct"], top1=False): 
        correct = []
        incorrect = []
        user_annot_data = self.annot_data[username]
        for image_id in self.labeled_images:
            user_label = user_annot_data[image_id]
            image = self._image_map[image_id]
            cur_image_labels = self.image_labels[image_id]
            filter_result = [f(image, cur_image_labels, user_label) for f in filters]
            if not filter_join(filter_result): 
                continue
            if top1:
                if user_label['label'] == self.wnid_map[image_id]:
                    correct.append(image_id)
                else:
                    incorrect.append(image_id)
            else:
                for state in correct_state:
                    if user_label["label"] in cur_image_labels[state]:
                        correct.append(image_id)
                        break
                else:
                    incorrect.append(image_id)
        return correct, incorrect

    def compute_accuracy(self, username, filters=[], filter_join=all, correct_state=["correct"], top1=False): 

        correct, incorrect = self.grade_predictions(username, filters, filter_join, correct_state, top1=top1)
        total = (len(incorrect) + len(correct))
        accuracy = len(correct)/ total

        return accuracy, total

def compute_acc_dict(h_a, username, extra_filters=[], top1=False):
    val_acc, _ = h_a.compute_accuracy(username, filters=[h_a.PROBLEMATIC_FILTER, h_a.VAL_SET] + extra_filters, top1=top1)
    v2_acc, _ = h_a.compute_accuracy(username, filters=[h_a.PROBLEMATIC_FILTER, h_a.VTWO_SET] + extra_filters, top1=top1)
    drop = val_acc - v2_acc
    return {"username": username, "val": round(val_acc, 4)*100, "v2": round(v2_acc, 4)*100}

