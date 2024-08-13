# def my_custom_loss(my_outputs, my_labels):
#     # specifying the batch size
#     my_batch_size = my_outputs.size()[0]
#
#     # calculating the log of softmax values
#     my_outputs = F.log_softmax(my_outputs, dim=1)
#
#     # selecting the values that correspond to labels
#     my_outputs = my_outputs[range(my_batch_size), my_labels]
#
#     return -torch.sum(my_outputs) / number_examples
#
#
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         intersection = (inputs * targets).sum()
#         dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#
#         return 1 - dice
