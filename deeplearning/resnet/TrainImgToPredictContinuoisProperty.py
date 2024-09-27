import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

from torch.optim.lr_scheduler import StepLR

Learning_Rate = 1e-4
width = height = 48  # image width and height
batchSize = 1


# ---------------------Create training image ---------------------------------------------------------
def ReadRandomImage():
    FillLevel = np.random.random()  # Set random fill level
    Img = np.zeros([48, 48, 3], np.uint8)  # Create black image
    Img[0:int(FillLevel * 48), :] = 255  # Fill the image with white up to FillLevel

    transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),
                               tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    Img = transformImg(Img)  # Transform to pytorch
    return Img, FillLevel


# --------------Load batch of images-----------------------------------------------------
def LoadBatch():  # Load batch of images
    images = torch.zeros([batchSize, 3, height, width])
    FillLevel = torch.zeros([batchSize])
    for i in range(batchSize):
        images[i], FillLevel[i] = ReadRandomImage()
    return images, FillLevel


# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place
Net = torchvision.models.resnet18(pretrained=True)  # Load net
Net.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)  # Change final layer to predict one value
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer
# scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.5,
    patience=1000,
    verbose=True,
    min_lr=1e-9,
)
# ----------------Train--------------------------------------------------------------------------
AverageLoss = np.ones([1000])  # Save average loss for display
for itr in range(500001):  # Training loop
    images, GTFillLevel = LoadBatch()  # Load taining batch
    images = torch.autograd.Variable(images, requires_grad=False).to(device)  # Load image
    GTFillLevel = torch.autograd.Variable(GTFillLevel, requires_grad=False).to(device)  # Load Ground truth fill level
    PredLevel = Net(images)  # make prediction
    Net.zero_grad()
    Loss = torch.abs(PredLevel - GTFillLevel).mean()
    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight
    AverageLoss[itr % 1000] = Loss.data.cpu().numpy()  # Save loss average
    avgloss = AverageLoss.mean()
    print(itr, ") Loss=", Loss.data.cpu().numpy(), 'AverageLoss', avgloss)  # Display loss
    if itr % 1000 == 0 and Loss < 0.0001:  # Save model
        print("Saving Model" + str(itr) + ".torch", "last lr:", scheduler.get_last_lr())  # Save model weight
        torch.save(Net.state_dict(), "./zoo/" + str(itr) + ".torch")

    if avgloss < 0.0001:
        print("Current lr rate:", scheduler.get_last_lr())
        break

    scheduler.step(avgloss)
