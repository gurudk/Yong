# import mmpose
import mmcv
import torch

print(torch.__version__)
# print(mmpose.__version__)
print(mmcv.__version__)

# from mmpose.apis import MMPoseInferencer
#
# img_path = './mmpose/tests/data/coco/000000000785.jpg'  # replace this with your own image path
#
# # instantiate the inferencer using the model alias
# inferencer = MMPoseInferencer('human')
#
# # The MMPoseInferencer API employs a lazy inference approach,
# # creating a prediction generator when given input
# result_generator = inferencer(img_path, show=True)
# result = next(result_generator)
