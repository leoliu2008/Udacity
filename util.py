import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torchvision import datasets, transforms, models

import pandas as pd
import numpy as np

from PIL import Image

from collections import OrderedDict

##two models with different input dimensions
arch={'vgg16':25088, 'densenet121': 1024}


def transform_load_image(data_dir):
    '''
    Argument: the directory of the data file
    return the data loaders for train/test/validation, plus the train data
    
    This function 
    1) locates the image files
    2) applies the transformations such as rotations, resize, flips and normalizations
    3)converts the images to tensor 
    4)the output also includes train_data, whose class_to_idx would be needed to save the checkpoint (ses 'save_checkpoint' funciton)
    
    '''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                        [0.229,0.224,0.225])
                                         ])

    validaiton_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                        [0.229,0.224,0.225])
                                           ])


    testing_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                        [0.229,0.224,0.225])
                                        ])
    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validaiton_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=testing_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = DataLoader(train_data, batch_size=64, shuffle = True)
    validloaders = DataLoader(valid_data, batch_size=64,shuffle = True)
    testloaders = DataLoader(test_data, batch_size=64, shuffle = True)
    return trainloaders, validloaders, testloaders,train_data


def construct_network(device, model_type='densenet121', drop_out=0.15, hidden_layer_1=512, hidden_layer_2=256, output_units = 102, learning_rate=0.005):
    '''
    Argument: 
    1)model type to be trained (densenet121 vs vgg16) and its associated input layer dimension. 
    2)dimension of hidden layers and output units
    3)other parameters such as drop_out and learning rate
    
    Return: model, loss function criterion, and optmizer, ready to use for model training
    
    This function
    1)rebuild the model classifier with the customised inputs from the model arugments
    2)use NLLLosss (negative log likelihood loss) to define the loss function
    3)use adam to optimize the parameter (e.g. update the weights during backpropagation), with predefined learning rate
    
    '''
    

    if model_type=='vgg16':
        model = models.vgg16(pretrained=True)
    elif model_type=='densenet121':
        model=models.densenet121(pretrained=True)
    else:
        print("the model you choose is not available, please choose vgg16 or densenet121")
    
    
    #frezze the model paratemers, as we only need to update the classifier parameter(e.g. weights for the input)
    
    for param in model.parameters():
        param.requires_grad = False



    input_units = arch[model_type]
    
    classifier = nn.Sequential(OrderedDict([ 
                                ('fc1',nn.Linear(input_units, hidden_layer_1)), 
                                ('relu1', nn.ReLU()),
                                ('do1',nn.Dropout(drop_out)), 
                                ('fc2', nn.Linear(hidden_layer_1,hidden_layer_2)), 
                                ('relu2', nn.ReLU()), 
                                ('do2',nn.Dropout(drop_out)), 
                                ('fc3', nn.Linear(hidden_layer_2,output_units)), 
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    #replace the pre-trained model's classifier withe the self-defined one as above
    model.classifier = classifier
    criterion = nn.NLLLoss()
    #only trian the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

   
    model.to(device)
    
    return model, criterion, optimizer


def test_network(model, criterion, optimizer, train_data, valid_data, device, epochs, print_every=40, steps=0):
    '''
    Argument: 
    1)model, criterion and optimizer from the 'construct_network' function
    2)train_data and valid_data from the 'transform_load_image' function
    3)other parameters such as epochs etc
    
    Return: print out the training loss, validation loss, and most importantly the accuracy of the model (with target 75% or above)
    
    This function
    1)use epochs for multiple rounds of model training,. ithin each epoch, it would kick off the model forward,  backpropagation and optmization of weights
    2)during model validaiton (with gradient turned off), it would turn the result of model output from log(probability) to probability
    3)it will use the class with highest probability as the prediction, and compare it to the label to calculate the accuracy
    
    
    '''
    
    model, criterion, optimizer = construct_network(device)
    
    model.to(device)


    for e in range (epochs): 
        running_loss = 0
        for inputs, labels in iter(train_data):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            #clear up the memory of the optimizer
            optimizer.zero_grad()

            #run the model forward
            outputs = model.forward(inputs)

            #define loss function
            loss = criterion(outputs, labels)

            #backpropagation
            loss.backward() 

            #update the weigth
            optimizer.step()

            #running total of loss value in scalar format
            running_loss += loss.item ()

            if steps % print_every == 0:
                #switch to model evaluation model to turn off drop out 
                model.eval ()

                # Turn off gradients for validation
                with torch.no_grad():
                    model.to (device)
                    valid_loss = 0
                    accuracy = 0
                    for inputs, labels in valid_data:

                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        valid_loss += criterion(output, labels).item()

                        ps = torch.exp(output)

                        #methhod 1 for accuracy calculation
                        #equality = (labels.data == ps.max(dim=1)[1])
                        #accuracy += equality.type(torch.FloatTensor).mean()

                        #methodl 2 for accuracy calculation
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_data)),
                      "Valid Accuracy: {:.3f}%.. ".format(accuracy/len(valid_data)*100))
                

                     

                running_loss = 0

                # Make sure training is back on
                model.train()

# # TODO: Do validation on the test set
# # Do validation on the test set
# correct = 0
# total = 0
# with torch.no_grad():
#     model.eval()
#     for data in trainloaders:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))                
                
                
                
                
                
def save_checkpoint(model, train_data, optimizer, model_pth_file, epochs):
    '''
    Argument: model & opitimizer,  train_data from the transform_load_image funciton, and the name of pth that we could like to serialize the modle
    Return: model to be saved to the pth file
    
    This function 
    1) uses train_data class_to_idx for the model 
    2) saves other featurs such as model_state_dict, classifier, optmizer_state_dict, epochs

    '''
    model.class_to_idx = train_data.class_to_idx
    model.to('cpu')
    checkpoint = {'model_state_dict': model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': train_data.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict,
              'epochs': epochs}

    torch.save(checkpoint, model_pth_file)
    
def load_checkpoint(model_pth_file,model_type='densenet121'):
    '''
    Argument: the name of the saved pth file, the modle type (e.g denssenet121 vs vgg16)
    Return: the fully loaded model with all parameters set up properly
    
    This function loads deep learning model checkpoint.
    
    '''
    
    # Load the saved file
    checkpoint = torch.load(model_pth_file)
    
    # Download pretrained model
    if model_type=='vgg16':
        model =models.vgg16(pretrained=True)
    elif model_type=='densenet121':
        model=models.densenet121(pretrained=True)
    else:
        print("the model you choose is not available, please choose vgg16 or densenet121")
     
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def process_image(image_path):
    ''' 
    Argument: the path of the image
    
    Return: the data of the image in the ndarray format
    
    the functions
    1)resizes, crops, and normalizes a PIL image
    2)converts the data from tensor to ndarray, ready for use in the imshow() funciton.
    
    '''
    image_pil=Image.open(image_path)
    transform=transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406],
                                                     [0.229,0.224,0.225])])
    
    image_tensor=transform(image_pil)
    #convert to numpy array
    image_ndarray=np.array(image_tensor)
    return image_ndarray
    # TODO: Process a PIL image for use in a PyTorch model
    
def imshow(image, ax=None, title=None):
    '''
    Argument; the image data in the ndarra format
    Return: display the image
    
    This functions
    1)transposes the image data(e.g move the color Channel RGB from 1st dimension to the 3rd dimension)
    2)undo the previous tranformation from process_image function (e.g "de"-normalisation)
    '''
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model_pth_file, cat_to_name, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    '''
    Argument; the locaiton of the image to be tested, the pth file for the saved model, the mapping on the category names and its indices, top K prediction
    Return: the top k prediced categoryes and its probability
    
    This functions
    1)loads the saved model from the pth file
    2)processes the image data and convert from ndarry to tensor, ready for model to train
    3)model to predict the top x classes and their probabilities (top_class, and top_p)
    4)map the index to get the category name (e.g the names of the flowers)
    '''
    
    model=load_checkpoint(model_pth_file)
    model.to(device)
    image_ndarray=process_image(image_path)
    image_tensor=torch.from_numpy(image_ndarray).type(torch.FloatTensor)

    image_tensor_added_dimension=image_tensor.unsqueeze(0)

    with torch.no_grad(): 
        output=model.forward(image_tensor_added_dimension.to(device))
    #         probability = F.softmax(output.data, dim=1)

        ps = torch.exp(output)

        top_p, top_class = ps.topk(top_k)
        
        top_p_list=np.array(top_p)[0]
        
        top_class_list=np.array(top_class)[0]

        #loading index and class mapping
        class_to_idx=model.class_to_idx
        #swap the location of index and class
        index_to_class = {x: y for y, x in class_to_idx.items()}
        
        top_labels=[index_to_class[x] for x in top_class_list]

        top_flowers=list(map(cat_to_name.get, top_labels))

    return top_p_list, top_flowers
