# Details

Date : 2025-04-06 17:17:41

Directory /media/jian/ssd4t/zero/zero/expAugmentation

Total : 113 files,  23174 codes, 1985 comments, 3752 blanks, all 28911 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [zero/expAugmentation/ObsProcessor/ObsProcessorBase.py](/zero/expAugmentation/ObsProcessor/ObsProcessorBase.py) | Python | 9 | 0 | 5 | 14 |
| [zero/expAugmentation/ObsProcessor/ObsProcessorDA3D.py](/zero/expAugmentation/ObsProcessor/ObsProcessorDA3D.py) | Python | 256 | 24 | 58 | 338 |
| [zero/expAugmentation/ObsProcessor/ObsProcessorPtv3.py](/zero/expAugmentation/ObsProcessor/ObsProcessorPtv3.py) | Python | 657 | 100 | 171 | 928 |
| [zero/expAugmentation/ReconLoss/ForwardKinematics.py](/zero/expAugmentation/ReconLoss/ForwardKinematics.py) | Python | 149 | 4 | 35 | 188 |
| [zero/expAugmentation/\_\_init\_\_.py](/zero/expAugmentation/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [zero/expAugmentation/config/DA3D.yaml](/zero/expAugmentation/config/DA3D.yaml) | YAML | 35 | 2 | 8 | 45 |
| [zero/expAugmentation/config/DA3D\_Original.yaml](/zero/expAugmentation/config/DA3D_Original.yaml) | YAML | 15 | 0 | 1 | 16 |
| [zero/expAugmentation/config/DP.yaml](/zero/expAugmentation/config/DP.yaml) | YAML | 88 | 1 | 15 | 104 |
| [zero/expAugmentation/config/FK.yaml](/zero/expAugmentation/config/FK.yaml) | YAML | 95 | 5 | 13 | 113 |
| [zero/expAugmentation/config/constants.py](/zero/expAugmentation/config/constants.py) | Python | 56 | 14 | 10 | 80 |
| [zero/expAugmentation/config/default.py](/zero/expAugmentation/config/default.py) | Python | 59 | 4 | 22 | 85 |
| [zero/expAugmentation/config/eval.yaml](/zero/expAugmentation/config/eval.yaml) | YAML | 28 | 3 | 9 | 40 |
| [zero/expAugmentation/config/expBase\_Lotus.yaml](/zero/expAugmentation/config/expBase_Lotus.yaml) | YAML | 114 | 16 | 14 | 144 |
| [zero/expAugmentation/config/expbase\_origin.yaml](/zero/expAugmentation/config/expbase_origin.yaml) | YAML | 116 | 35 | 21 | 172 |
| [zero/expAugmentation/dataset/dataset\_DA3D.py](/zero/expAugmentation/dataset/dataset_DA3D.py) | Python | 242 | 25 | 52 | 319 |
| [zero/expAugmentation/dataset/dataset\_DP\_use\_obsprocessor.py](/zero/expAugmentation/dataset/dataset_DP_use_obsprocessor.py) | Python | 119 | 11 | 28 | 158 |
| [zero/expAugmentation/dataset/dataset\_FK.py](/zero/expAugmentation/dataset/dataset_FK.py) | Python | 279 | 30 | 68 | 377 |
| [zero/expAugmentation/dataset/dataset\_expbase\_DP.py](/zero/expAugmentation/dataset/dataset_expbase_DP.py) | Python | 345 | 56 | 79 | 480 |
| [zero/expAugmentation/dataset/dataset\_expbase\_voxel\_augment \_use\_perceptor.py](/zero/expAugmentation/dataset/dataset_expbase_voxel_augment%20_use_perceptor.py) | Python | 346 | 71 | 88 | 505 |
| [zero/expAugmentation/dataset/dataset\_expbase\_voxel\_augment.py](/zero/expAugmentation/dataset/dataset_expbase_voxel_augment.py) | Python | 345 | 72 | 89 | 506 |
| [zero/expAugmentation/dataset/dataset\_origin.py](/zero/expAugmentation/dataset/dataset_origin.py) | Python | 387 | 43 | 78 | 508 |
| [zero/expAugmentation/eval\_DP\_PTV3.py](/zero/expAugmentation/eval_DP_PTV3.py) | Python | 337 | 36 | 82 | 455 |
| [zero/expAugmentation/eval\_LongHorizon.py](/zero/expAugmentation/eval_LongHorizon.py) | Python | 481 | 42 | 107 | 630 |
| [zero/expAugmentation/eval\_expbase.py](/zero/expAugmentation/eval_expbase.py) | Python | 490 | 38 | 102 | 630 |
| [zero/expAugmentation/models/Base/BaseAll.py](/zero/expAugmentation/models/Base/BaseAll.py) | Python | 41 | 0 | 13 | 54 |
| [zero/expAugmentation/models/DAFC/Policy.py](/zero/expAugmentation/models/DAFC/Policy.py) | Python | 484 | 88 | 101 | 673 |
| [zero/expAugmentation/models/DAFC/components/clip.py](/zero/expAugmentation/models/DAFC/components/clip.py) | Python | 34 | 1 | 9 | 44 |
| [zero/expAugmentation/models/DAFC/components/layers.py](/zero/expAugmentation/models/DAFC/components/layers.py) | Python | 415 | 15 | 60 | 490 |
| [zero/expAugmentation/models/DAFC/components/multihead\_custom\_attention.py](/zero/expAugmentation/models/DAFC/components/multihead_custom_attention.py) | Python | 401 | 18 | 49 | 468 |
| [zero/expAugmentation/models/DAFC/components/position\_encodings.py](/zero/expAugmentation/models/DAFC/components/position_encodings.py) | Python | 112 | 0 | 32 | 144 |
| [zero/expAugmentation/models/DAFC/components/resnet.py](/zero/expAugmentation/models/DAFC/components/resnet.py) | Python | 49 | 1 | 11 | 61 |
| [zero/expAugmentation/models/DiffuserActor3D/Policy.py](/zero/expAugmentation/models/DiffuserActor3D/Policy.py) | Python | 487 | 88 | 107 | 682 |
| [zero/expAugmentation/models/DiffuserActor3D/Policy\_significant\_change.py](/zero/expAugmentation/models/DiffuserActor3D/Policy_significant_change.py) | Python | 487 | 88 | 107 | 682 |
| [zero/expAugmentation/models/DiffuserActor3D/components/clip.py](/zero/expAugmentation/models/DiffuserActor3D/components/clip.py) | Python | 34 | 1 | 9 | 44 |
| [zero/expAugmentation/models/DiffuserActor3D/components/layers.py](/zero/expAugmentation/models/DiffuserActor3D/components/layers.py) | Python | 415 | 15 | 58 | 488 |
| [zero/expAugmentation/models/DiffuserActor3D/components/multihead\_custom\_attention.py](/zero/expAugmentation/models/DiffuserActor3D/components/multihead_custom_attention.py) | Python | 401 | 18 | 49 | 468 |
| [zero/expAugmentation/models/DiffuserActor3D/components/position\_encodings.py](/zero/expAugmentation/models/DiffuserActor3D/components/position_encodings.py) | Python | 112 | 0 | 32 | 144 |
| [zero/expAugmentation/models/DiffuserActor3D/components/resnet.py](/zero/expAugmentation/models/DiffuserActor3D/components/resnet.py) | Python | 49 | 1 | 11 | 61 |
| [zero/expAugmentation/models/DiffuserActor3D\_original.py/policy.py](/zero/expAugmentation/models/DiffuserActor3D_original.py/policy.py) | Python | 553 | 49 | 68 | 670 |
| [zero/expAugmentation/models/FK/Policy.py](/zero/expAugmentation/models/FK/Policy.py) | Python | 342 | 46 | 110 | 498 |
| [zero/expAugmentation/models/FK/component/ddpm.py](/zero/expAugmentation/models/FK/component/ddpm.py) | Python | 39 | 1 | 14 | 54 |
| [zero/expAugmentation/models/FK/component/ddpm\_copy.py](/zero/expAugmentation/models/FK/component/ddpm_copy.py) | Python | 77 | 3 | 22 | 102 |
| [zero/expAugmentation/models/FK/component/test.py](/zero/expAugmentation/models/FK/component/test.py) | Python | 5 | 0 | 4 | 9 |
| [zero/expAugmentation/models/clean/PointTransformerV3/model.py](/zero/expAugmentation/models/clean/PointTransformerV3/model.py) | Python | 946 | 56 | 107 | 1,109 |
| [zero/expAugmentation/models/clean/PointTransformerV3/model\_ca.py](/zero/expAugmentation/models/clean/PointTransformerV3/model_ca.py) | Python | 399 | 18 | 39 | 456 |
| [zero/expAugmentation/models/clean/PointTransformerV3/serialization/\_\_init\_\_.py](/zero/expAugmentation/models/clean/PointTransformerV3/serialization/__init__.py) | Python | 8 | 0 | 1 | 9 |
| [zero/expAugmentation/models/clean/PointTransformerV3/serialization/default.py](/zero/expAugmentation/models/clean/PointTransformerV3/serialization/default.py) | Python | 46 | 1 | 13 | 60 |
| [zero/expAugmentation/models/clean/PointTransformerV3/serialization/hilbert.py](/zero/expAugmentation/models/clean/PointTransformerV3/serialization/hilbert.py) | Python | 190 | 42 | 72 | 304 |
| [zero/expAugmentation/models/clean/PointTransformerV3/serialization/z\_order.py](/zero/expAugmentation/models/clean/PointTransformerV3/serialization/z_order.py) | Python | 96 | 6 | 25 | 127 |
| [zero/expAugmentation/models/clean/\_\_init\_\_.py](/zero/expAugmentation/models/clean/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [zero/expAugmentation/models/clean/base.py](/zero/expAugmentation/models/clean/base.py) | Python | 64 | 0 | 16 | 80 |
| [zero/expAugmentation/models/clean/optim/\_\_init\_\_.py](/zero/expAugmentation/models/clean/optim/__init__.py) | Python | 6 | 0 | 2 | 8 |
| [zero/expAugmentation/models/clean/optim/adamw.py](/zero/expAugmentation/models/clean/optim/adamw.py) | Python | 81 | 13 | 20 | 114 |
| [zero/expAugmentation/models/clean/optim/lookahead.py](/zero/expAugmentation/models/clean/optim/lookahead.py) | Python | 81 | 7 | 10 | 98 |
| [zero/expAugmentation/models/clean/optim/misc.py](/zero/expAugmentation/models/clean/optim/misc.py) | Python | 49 | 1 | 7 | 57 |
| [zero/expAugmentation/models/clean/optim/radam.py](/zero/expAugmentation/models/clean/optim/radam.py) | Python | 148 | 4 | 58 | 210 |
| [zero/expAugmentation/models/clean/optim/ralamb.py](/zero/expAugmentation/models/clean/optim/ralamb.py) | Python | 71 | 6 | 23 | 100 |
| [zero/expAugmentation/models/clean/optim/rangerlars.py](/zero/expAugmentation/models/clean/optim/rangerlars.py) | Python | 8 | 3 | 4 | 15 |
| [zero/expAugmentation/models/clean/optim/sched.py](/zero/expAugmentation/models/clean/optim/sched.py) | Python | 95 | 2 | 17 | 114 |
| [zero/expAugmentation/models/clean/test\_model\_expbase.py](/zero/expAugmentation/models/clean/test_model_expbase.py) | Python | 220 | 19 | 49 | 288 |
| [zero/expAugmentation/models/clean/utils/action\_position\_utils.py](/zero/expAugmentation/models/clean/utils/action_position_utils.py) | Python | 68 | 36 | 13 | 117 |
| [zero/expAugmentation/models/clean/utils/point\_cloud.py](/zero/expAugmentation/models/clean/utils/point_cloud.py) | Python | 34 | 1 | 13 | 48 |
| [zero/expAugmentation/models/clean/utils/rlbench\_keystep\_detection.py](/zero/expAugmentation/models/clean/utils/rlbench_keystep_detection.py) | Python | 37 | 1 | 10 | 48 |
| [zero/expAugmentation/models/clean/utils/robot\_box.py](/zero/expAugmentation/models/clean/utils/robot_box.py) | Python | 73 | 0 | 16 | 89 |
| [zero/expAugmentation/models/clean/utils/rotation\_transform.py](/zero/expAugmentation/models/clean/utils/rotation_transform.py) | Python | 158 | 9 | 30 | 197 |
| [zero/expAugmentation/models/dp2d/components/PointTransformerV3/model.py](/zero/expAugmentation/models/dp2d/components/PointTransformerV3/model.py) | Python | 946 | 56 | 107 | 1,109 |
| [zero/expAugmentation/models/dp2d/components/PointTransformerV3/model\_ca.py](/zero/expAugmentation/models/dp2d/components/PointTransformerV3/model_ca.py) | Python | 399 | 18 | 39 | 456 |
| [zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/\_\_init\_\_.py](/zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/__init__.py) | Python | 8 | 0 | 1 | 9 |
| [zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/default.py](/zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/default.py) | Python | 46 | 1 | 13 | 60 |
| [zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/hilbert.py](/zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/hilbert.py) | Python | 190 | 42 | 72 | 304 |
| [zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/z\_order.py](/zero/expAugmentation/models/dp2d/components/PointTransformerV3/serialization/z_order.py) | Python | 96 | 6 | 25 | 127 |
| [zero/expAugmentation/models/dp2d/components/diffusion/conditional\_unet1d.py](/zero/expAugmentation/models/dp2d/components/diffusion/conditional_unet1d.py) | Python | 182 | 10 | 37 | 229 |
| [zero/expAugmentation/models/dp2d/components/diffusion/conditionalunet2d.py](/zero/expAugmentation/models/dp2d/components/diffusion/conditionalunet2d.py) | Python | 183 | 8 | 45 | 236 |
| [zero/expAugmentation/models/dp2d/components/diffusion/conv2d\_components.py](/zero/expAugmentation/models/dp2d/components/diffusion/conv2d_components.py) | Python | 75 | 9 | 31 | 115 |
| [zero/expAugmentation/models/dp2d/components/diffusion/pe.py](/zero/expAugmentation/models/dp2d/components/diffusion/pe.py) | Python | 15 | 0 | 4 | 19 |
| [zero/expAugmentation/models/dp2d/ptv3\_DP1d\_policy.py](/zero/expAugmentation/models/dp2d/ptv3_DP1d_policy.py) | Python | 163 | 28 | 51 | 242 |
| [zero/expAugmentation/models/lotus/PointTransformerV3/model.py](/zero/expAugmentation/models/lotus/PointTransformerV3/model.py) | Python | 946 | 56 | 107 | 1,109 |
| [zero/expAugmentation/models/lotus/PointTransformerV3/model\_ca.py](/zero/expAugmentation/models/lotus/PointTransformerV3/model_ca.py) | Python | 399 | 18 | 39 | 456 |
| [zero/expAugmentation/models/lotus/PointTransformerV3/serialization/\_\_init\_\_.py](/zero/expAugmentation/models/lotus/PointTransformerV3/serialization/__init__.py) | Python | 8 | 0 | 1 | 9 |
| [zero/expAugmentation/models/lotus/PointTransformerV3/serialization/default.py](/zero/expAugmentation/models/lotus/PointTransformerV3/serialization/default.py) | Python | 46 | 1 | 13 | 60 |
| [zero/expAugmentation/models/lotus/PointTransformerV3/serialization/hilbert.py](/zero/expAugmentation/models/lotus/PointTransformerV3/serialization/hilbert.py) | Python | 190 | 42 | 72 | 304 |
| [zero/expAugmentation/models/lotus/PointTransformerV3/serialization/z\_order.py](/zero/expAugmentation/models/lotus/PointTransformerV3/serialization/z_order.py) | Python | 96 | 6 | 25 | 127 |
| [zero/expAugmentation/models/lotus/\_\_init\_\_.py](/zero/expAugmentation/models/lotus/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [zero/expAugmentation/models/lotus/assets/task\_new\_keystep\_ids.json](/zero/expAugmentation/models/lotus/assets/task_new_keystep_ids.json) | JSON | 113 | 0 | 1 | 114 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_instructions\_new.json](/zero/expAugmentation/models/lotus/assets/taskvars_instructions_new.json) | JSON | 594 | 0 | 0 | 594 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_instructions\_peract.json](/zero/expAugmentation/models/lotus/assets/taskvars_instructions_peract.json) | JSON | 1,583 | 0 | 0 | 1,583 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_peract.json](/zero/expAugmentation/models/lotus/assets/taskvars_peract.json) | JSON | 251 | 0 | 0 | 251 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_target\_label\_zrange.json](/zero/expAugmentation/models/lotus/assets/taskvars_target_label_zrange.json) | JSON | 1,968 | 0 | 3 | 1,971 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_test\_l2.json](/zero/expAugmentation/models/lotus/assets/taskvars_test_l2.json) | JSON | 30 | 0 | 1 | 31 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_test\_l3.json](/zero/expAugmentation/models/lotus/assets/taskvars_test_l3.json) | JSON | 23 | 0 | 0 | 23 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_test\_l4.json](/zero/expAugmentation/models/lotus/assets/taskvars_test_l4.json) | JSON | 14 | 0 | 0 | 14 |
| [zero/expAugmentation/models/lotus/assets/taskvars\_train.json](/zero/expAugmentation/models/lotus/assets/taskvars_train.json) | JSON | 33 | 0 | 0 | 33 |
| [zero/expAugmentation/models/lotus/base.py](/zero/expAugmentation/models/lotus/base.py) | Python | 64 | 0 | 16 | 80 |
| [zero/expAugmentation/models/lotus/model\_expbase.py](/zero/expAugmentation/models/lotus/model_expbase.py) | Python | 604 | 46 | 78 | 728 |
| [zero/expAugmentation/models/lotus/optim/\_\_init\_\_.py](/zero/expAugmentation/models/lotus/optim/__init__.py) | Python | 6 | 0 | 2 | 8 |
| [zero/expAugmentation/models/lotus/optim/adamw.py](/zero/expAugmentation/models/lotus/optim/adamw.py) | Python | 81 | 13 | 20 | 114 |
| [zero/expAugmentation/models/lotus/optim/lookahead.py](/zero/expAugmentation/models/lotus/optim/lookahead.py) | Python | 81 | 7 | 10 | 98 |
| [zero/expAugmentation/models/lotus/optim/misc.py](/zero/expAugmentation/models/lotus/optim/misc.py) | Python | 49 | 1 | 7 | 57 |
| [zero/expAugmentation/models/lotus/optim/radam.py](/zero/expAugmentation/models/lotus/optim/radam.py) | Python | 148 | 4 | 58 | 210 |
| [zero/expAugmentation/models/lotus/optim/ralamb.py](/zero/expAugmentation/models/lotus/optim/ralamb.py) | Python | 71 | 6 | 23 | 100 |
| [zero/expAugmentation/models/lotus/optim/rangerlars.py](/zero/expAugmentation/models/lotus/optim/rangerlars.py) | Python | 8 | 3 | 4 | 15 |
| [zero/expAugmentation/models/lotus/optim/sched.py](/zero/expAugmentation/models/lotus/optim/sched.py) | Python | 95 | 2 | 17 | 114 |
| [zero/expAugmentation/models/lotus/policy.py](/zero/expAugmentation/models/lotus/policy.py) | Python | 0 | 0 | 1 | 1 |
| [zero/expAugmentation/models/lotus/utils/action\_position\_utils.py](/zero/expAugmentation/models/lotus/utils/action_position_utils.py) | Python | 68 | 120 | 24 | 212 |
| [zero/expAugmentation/models/lotus/utils/point\_cloud.py](/zero/expAugmentation/models/lotus/utils/point_cloud.py) | Python | 34 | 1 | 13 | 48 |
| [zero/expAugmentation/models/lotus/utils/rlbench\_keystep\_detection.py](/zero/expAugmentation/models/lotus/utils/rlbench_keystep_detection.py) | Python | 37 | 1 | 10 | 48 |
| [zero/expAugmentation/models/lotus/utils/robot\_box.py](/zero/expAugmentation/models/lotus/utils/robot_box.py) | Python | 45 | 35 | 18 | 98 |
| [zero/expAugmentation/models/lotus/utils/rotation\_transform.py](/zero/expAugmentation/models/lotus/utils/rotation_transform.py) | Python | 158 | 9 | 30 | 197 |
| [zero/expAugmentation/trainer\_DA3D.py](/zero/expAugmentation/trainer_DA3D.py) | Python | 100 | 30 | 35 | 165 |
| [zero/expAugmentation/trainer\_DP.py](/zero/expAugmentation/trainer_DP.py) | Python | 119 | 30 | 36 | 185 |
| [zero/expAugmentation/trainer\_FK.py](/zero/expAugmentation/trainer_FK.py) | Python | 102 | 32 | 35 | 169 |
| [zero/expAugmentation/trainer\_expbase.py](/zero/expAugmentation/trainer_expbase.py) | Python | 243 | 54 | 58 | 355 |
| [zero/expAugmentation/visualization.py](/zero/expAugmentation/visualization.py) | Python | 1 | 0 | 1 | 2 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)