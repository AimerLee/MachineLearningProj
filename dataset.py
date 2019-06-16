import os
import re
from typing import List

from BCMI.constant import ATTRS
from BCMI.dataset import Resolver


class CustomResolver(Resolver):
    def __init__(self, dataset_dir, labels):
        super().__init__(dataset_dir)
        self.labels = labels

    def resolve(self) -> List[dict]:
        """
        支持的目录结构为：
               - Dataset
                   - Subject
                       - Experiment
                           - Trial
        如果使用不同的目录结构，可以继承此类并重载此函数，实现dataset的index
        """
        rows = []
        for subject_name in os.listdir(self.dataset_dir):
            subject_dir = os.path.join(self.dataset_dir, subject_name)
            for session_name in os.listdir(subject_dir):
                session_dir = os.path.join(subject_dir, session_name)
                for trial_name in os.listdir(session_dir):
                    # Store the dataset directory and the relative path separately.
                    trial_path = os.path.join(session_dir, trial_name)
                    session_index = int(session_name)

                    # Obtain feature name, trial index
                    re_result = re.match(r'([a-zA-Z]+)_([0-9]+)\.npy', trial_name)
                    if re_result is None:
                        raise Exception("Format Wrong: {:s}".format(trial_name))
                    feature_type = re_result.group(1)
                    trial_index = int(re_result.group(2))

                    label = self.labels[session_index][trial_index]

                    row = {ATTRS.SUBJECT_NAME: subject_name,
                           ATTRS.SESSION_INDEX: session_index,
                           ATTRS.TRIAL_INDEX: trial_index,
                           ATTRS.DATA_PATH: os.path.normpath(trial_path),
                           ATTRS.LABEL: label,
                           ATTRS.FEATURE_TYPE: feature_type}
                    rows.append(row)
        return rows
