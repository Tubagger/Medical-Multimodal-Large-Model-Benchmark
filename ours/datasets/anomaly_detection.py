from typing import Optional, Sequence, Dict
from ours.datasets.base import BaseDataset
from ours.methods.base import BaseMethod
from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType

import yaml
import json
import os


@registry.register_dataset()
class AnomalyData(BaseDataset):

    dataset_ids: Sequence[str] = [
        "anomaly-detection-chexpert",
        "anomaly-detection-isic",
        "anomaly-detection-brain",
        "anomaly-detection-mura",
        "anomaly-detection-oct"
    ]


    dataset_config = {

        "anomaly-detection-chexpert":
        "./ours/configs/datasets/anomaly-detection-chexpert.yaml",

        "anomaly-detection-isic":
        "./ours/configs/datasets/anomaly-detection-isic.yaml",

        "anomaly-detection-brain":
        "./ours/configs/datasets/anomaly-detection-brain.yaml",

        "anomaly-detection-mura":
        "./ours/configs/datasets/anomaly-detection-mura.yaml",

        "anomaly-detection-oct":
        "./ours/configs/datasets/anomaly-detection-oct.yaml"
    }



    def __init__(
        self,
        dataset_id: str,
        model_id: str,
        method_hook: Optional[BaseMethod] = None,
        debug_mode: bool = False,
        **kwargs
    ):

        super().__init__(
            dataset_id=dataset_id,
            model_id=model_id,
            method_hook=method_hook,
            debug_mode=debug_mode
        )


        # ==========================
        # Load config
        # ==========================

        with open(
            self.dataset_config[dataset_id]
        ) as f:

            self.config = yaml.load(
                f,
                Loader=yaml.FullLoader
            )


        self.image_dir = self.config["image_dir"]

        self.label_file = self.config["annotation_file"]


        if debug_mode:
            self.nums = self.config.get(
                "debug_nums",
                None
            )

        else:
            self.nums = self.config.get(
                "nums",
                None
            )


        assert os.path.exists(
            self.image_dir
        ), self.image_dir


        assert os.path.exists(
            self.label_file
        ), self.label_file



        # ==========================
        # Load json
        # ==========================

        with open(
            self.label_file,
            "r",
            encoding="utf-8"
        ) as f:

            samples=json.load(f)



        if self.nums is not None:

            samples=samples[:self.nums]



        # ==========================
        # Build dataset
        # ==========================

        dataset=[]


        for sample in samples:


            id=sample["id"]





            image_name=sample["image"]


            image_path=os.path.join(
                self.image_dir,
                image_name
            )


            assert os.path.exists(
                image_path
            ), image_path



            # ======================
            # New MCQ prompt
            # ======================

            question = sample["question"]

            options = sample["options"]


            PROMPT=f"""
            You are a medical AI assistant.
            {question}
            {options}
            Please select the correct option.
            Output ONLY the option letter.
            For example:
            A
            """



            dataset.append(

                ImageTxtSample(

                    id=id,

                    image_path=image_path,

                    text=PROMPT,

                    target=
                        sample["answer"]
                )

            )


        self.dataset=dataset



    def __getitem__(
        self,
        index:int
    )->_OutputType:


        if self.method_hook:

            return self.method_hook.run(
                self.dataset[index]
            )


        return self.dataset[index]



    def __len__(self):

        return len(self.dataset)
