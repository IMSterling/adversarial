from yolov3.modified_detect import *
from yolov3.utils.simple_dataset import *
from yolov3.utils.utils import *
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd import Variable
import cv2
import os
import gin
import logging
import sys
import random
import datetime
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
"""
--------------------------------------------------
Summer Research 2020
Ian McDiarmid-Sterling 
Swarthmore College

Welcome to Ian McDiarmid-Sterling's Adversarial attack generator! Note --> the functions detect and fgsm_attack are direct implementations of someone else's code that I have
mearly modified for use here. Enjoy, and feel free to email me at ianmcdiarmidsterling@gmail.com with questions. 
--------------------------------------------------

"""

@gin.configurable()
def apply_patch(positions,patched_img, adv_patch,givenX=None,givenY=None,randomize=True):
    """
    This function applies a adversarial patch (delta) to a image tensor, returning the combined tensor
    Params:
       positions --> A list of tuples (x,y) of allowable locations for patch placement
       patched_img --> A torch.tensor of an image
       adv_patch --> A torch.tensor of the adversarial patch
       givenX --> X value to place patch when using randomize = False
       givenY --> Y value to place patch when using randomize = False
       truths --> A torch.tensor containing the bounding box coordinates for the detection
       randomize --> The option to place delta at a random location or a specified location (boolean) : When using randomize = False, givenX and givenY must be supplied

    Returns:
        patched_img --> A torch.Tensor of the combined delta and image

    """
    # Check to make sure positions is not empty
    if randomize ==True:
        assert (len(positions) !=0),"No random positions to choose from: change offset or patch size"

    # Check to make sure all inputs are correct
    assert randomize == True or (givenX!=None and givenY!=None), 'Specify coordinates for patch location when using randomize = False'

    # Name some variables for convinience
    img_height = patched_img.shape[1]
    img_width = patched_img.shape[2]
    patch_height = adv_patch.shape[1]
    patch_width = adv_patch.shape[2]

    if randomize == True:

        # Pick a random (x,y) from positions
        x0,y0 = random.sample(positions,1)[0]

        # Calculate the padding for that x0,y0
        tpad = y0
        bpad = img_height-(patch_height+y0)
        lpad = x0
        rpad = img_width-(patch_width+x0)


    elif randomize == False:

        # set (x0,y0) to the givenX and givenY
        x0,y0 = givenX,givenY

        # Calculate the padding for that x0,y0
        tpad = y0
        bpad = img_height - (patch_height + y0)
        lpad = x0
        rpad = img_width - (patch_width + x0)



    # Perform Padding
    mypad = nn.ConstantPad2d((lpad,rpad,tpad,bpad), 0)
    padded_patch = mypad(adv_patch)

    # Combine padded patch and original image
    patched_img = patched_img + padded_patch

    # Return patched_img
    return patched_img


def calculate_positions(patched_img, adv_patch, truths,offset):
    """
    This function calculates a list of every possible (x,y) a patch could be placed on an image
    Params:
        patched_img --> A torch.tensor of an image
        adv_patch --> A torch.tensor of the adversarial patch
        truths --> A torch.tensor containing the bounding box coordinates for the detection
        offset --> The minimum distance between the bounding box and the edge of the patch (int)

    Returns:
        locations --> A list of every tuple (x,y) the patch can be placed (list)

    """

    #Gather measurements for ease
    img_height = patched_img.shape[1]
    img_width = patched_img.shape[2]
    patch_height = adv_patch.shape[1]
    patch_width = adv_patch.shape[2]


    #Convert bounding box coordinates to list of form (x1,y1,x2,y2) for the  bounding box
    object_bounding_box = truths.int().cpu().numpy().tolist()

    locations = []

    #This block adds all locations to the sides of the bounding box to the list locations
    vertical_y= range(0,img_height-patch_height)
    left = set(range(max((object_bounding_box[0]-patch_width-offset),0),(object_bounding_box[0]-patch_width)))
    right = set(range(object_bounding_box[2],min((img_width-patch_width,object_bounding_box[2]+offset))))
    allowableX = left.union(right)
    for y in vertical_y:
        for x in allowableX:
            locations.append((x,y))

    # This block adds all locations above and below the bounding box to the list locations
    horizontal_x = range(0,img_width-patch_width)
    top = set(range(max((object_bounding_box[1]-patch_height-offset),0),(object_bounding_box[1]-patch_height)))
    bottom = set(range(object_bounding_box[3],min((img_height-patch_height,object_bounding_box[3]+offset))))
    allowableY = top.union(bottom)
    for x in horizontal_x:
        for y in allowableY:
            locations.append((x,y))

    #return the list of possible locations for the patch
    return(locations)


def fgsm_attack(delta, epsilon, data_grad):
    """
    This function modifies a delta by epsilon * sign of data_grad
    Params:
        delta --> A torch.tensor of the adversarial patch
        epsilon --> The amount to modify delta (float)
        data_grad --> The current gradient direction (torch.tensor)

    Returns:
        perturbed_delta --> The new delta (torch.tensor)

    """



    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed delta by adjusting each pixel of the input delta
    perturbed_delta = delta + epsilon*sign_data_grad

    # Return the perturbed delta
    return perturbed_delta


def load_image(path,height,width):
    """
    This function loads a series of images as torch.tensors given the path to the directory of images
    Params:
       path --> the path to the directory of images (string)
       height --> The desired height of the image (int)
       width --> The desired width of the image (int)

    Returns:
        image_tensor --> A torch.tensor of the image data
        image_names  --> list of image names

    """
    #Check the path exists
    if not os.path.exists(path):
        print("Hey! Image path {} doesn't exist".format(path))
        sys.exit()

    #Concatinate all images into one numpy array
    first = True
    image_names = []
    for file in os.listdir(path):
        if file == '.DS_Store': # Remove .DS_Store files that appear in unix (Macbooks)
            continue
        image_names.append(file)
        image = cv2.imread(os.path.join(path,file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Perform color conversion from BGR to RBG
        image = cv2.resize(image,(width,height),interpolation=cv2.INTER_NEAREST) # Resize the image to the desired dimensions
        image = image[np.newaxis, :, :, :] # Add a batch axis for concatination
        if first == True:
            first = False
            saved = image
        else:
            saved = np.concatenate((saved,image),axis=0)

    # Convert the numpy array to a torch.tensor and permute so the final tensor has form Image x Channels x Height x Width
    image_tensor = torch.from_numpy(saved)
    image_tensor = image_tensor.permute(0,3,1,2)

    # Reutrn the final tensor and list of names
    return image_tensor, image_names


@gin.configurable
def generate_attack(names,writer,model, device, run_name, image_path, load_path, epochs,  load, lr,
                     randomize, num_random_generations, patch_width, patch_height,height,width,offset,lr_scale,patience,givenX=None,givenY=None,seed=152,
                    lr_length=1e6,alpha=1.0, return_output=False):
    """
      This function generates an adversarial attack for a series of images
      Params:
         names --> The list of names associated with yoloV3's returned class label (list)
         writer --> A tensorboard logging object used to log data (object)
         model --> The Nnet model (darknet model)
         device --> The device to send torch.tensors to (either 'cpu' or 'cuda:0')
         run_name --> The name of this run (string)
         image_path --> The path to the data directory for the images (string)
         load_path--> The path to load data from (the previous path the saved .npy file) (string)
         epochs --> The desired number of epochs to run for
         load --> The decision to load a previous delta and saved state or start fresh (boolean)
         lr --> The starting learning rate (float)
         randomize --> The decision to randomize location when generating the patch (boolean)
         num_random_generations --> The number of random generations to perform, per image, per epoch, to obtain a smoother gradient (int)
         patch_width --> The width of the adversarial patch (int)
         patch_height --> The height of the adversarial patch (int)
         offset --> The minimum distance between the bounding box and the edge of the patch (int)
         lr_scale --> The factor to reduce learning rate by every lr_length epochs (flaot)
         patience --> The number of epochs to continue for without finding a new lowest confidence (int)
         givenX --> Optional parameter for use with randomize = False for the desired x location of the delta (OPTIONAL: int)
         givenY --> Optional parameter for use with randomize = False for the desired y location of the delta (OPTIONAL: int)
         seed --> The random seed for all random number generation (int)
         lr_length --> Reduce learning rate bt lr_scale every lr_length epochs (int)
         alpha --> Momentum (float)
         return_output --> Flag to return output (boolean)
      Returns:
         confidence --> OPTIONAL (if return_output = True) reutrns best confidence during training (float)

       """


    # Print start info to console
    start_time  = datetime.datetime.now()
    logging.info('--------------------------------------------------------------------------------------------')
    logging.info('------------------- Attack Generation started: {}   -----------------------'.format(start_time.strftime("%Y.%m.%d:%H.%M.%S")))
    logging.info('Running on {} with seed: {} and randomize: {}'.format(device, seed,randomize))

    # Create the result_path to save to
    result_path = os.path.join('results', run_name)

    # Load images as torch.tensor of (images x channels x height x width)
    image_tensor,_ = load_image(image_path,height,width)

    # Perform forward pass to get ground truth predictions on all imagery
    pred = detect(image_tensor,model,device)

    # Initialize variables
    gt_confidence = []
    gt_box = []

    # For each prediction ...
    for i in range(len(pred)):
        if return_output == False:
            logging.info("Found a {} with {:5.7f} % certainty on initial image load of image {}".format(names[int(pred[i][0][5].item())], pred[i][0][4].item(),i+1))

        # Add confidence to list of confidences
        gt_confidence.append(torch.tensor(pred[i][0][4].item()))

        # Extract ground truth detection box
        ground_truth = pred[i][0][0:4]

        # Add detection bounding box to the list of detection bounding boxes
        gt_box.append(ground_truth)

    if load == True:
        logging.info('Loading patch...')

        # Load delta
        delta = torch.from_numpy(np.load(load_path))
        delta = Variable(delta, requires_grad=True)

        # Calculate the path to the state dictionary
        state_load_path = load_path.replace('numpy_files','state_files')
        state_load_path = state_load_path.replace('.npy', '.csv')

        # Attempt to load the state dictionary associated with the delta
        try:
            meta_data = pd.read_csv(state_load_path)
            load_epoch = int(meta_data.epoch[0])
            best_epoch = int(meta_data.best_epoch[0])
            loss = float(meta_data.loss[0])
            confidence = float(meta_data.confidence[0])
            best_confidence = float(meta_data.best_confidence[0])
        except:
            logging.info('No associated state dictionary with that numpy file ')

        # Add one point to tensorboard so plotting is continous
        if writer != None and return_output == False:
            writer.add_scalar('LearningRate',float(lr), load_epoch)
            writer.add_scalar('Loss',float(loss), load_epoch)
            writer.add_scalar('Confidence',float(confidence), load_epoch)


    else:

        # Create a new delta of the specified dimensions
        delta = torch.ones((3,patch_width,patch_height), requires_grad=True)
        best_confidence = 1
        load_epoch = -1

    # Initialize the list of positions
    positions = []

    # For each image, add the acceptable positions for random generaton to the list positions
    for i in range(image_tensor.shape[0]):
        truths = gt_box[i] # Isolate a single image's detection box
        single_tensor = image_tensor[i, :, :, :] # Isolate a single image
        image_locations = calculate_positions(single_tensor, delta, truths,offset) # Calculate acceptable positions for the single image
        logging.info("Calculated a possible {} random positions for patch placement on image {}".format(len(image_locations),i+1))
        positions.append(image_locations)

    #debug = False
    # Allow error detection during backpropagation
    torch.autograd.set_detect_anomaly(True)

    # No last gradient to use
    last_grad = None

    # Optimization loop
    for t in range(load_epoch+1,load_epoch+epochs+1):

        # Learning rate scheduler
        if t and t % lr_length == 0:
            lr = np.max((5.0, lr*lr_scale))  # Prevent learning rate dropping below 5
            if return_output == False:
                logging.info("Adjusting learning rate to {}".format(lr))

        if randomize == True:

            # Initialize variables
            sum_grad = 0
            sum_confidence = 0
            sum_loss = 0
            num_finished = 0

            # For every random generation ...
            for i in range(num_random_generations):
                first = True
                for j in range(image_tensor.shape[0]):

                    # Isolate a single image tensor
                    single_tensor = image_tensor[j,:,:,:]
                    # Isolate the positions for that image
                    image_locations = positions[j]
                    # Combine image and delta
                    modified_image = apply_patch(patched_img=single_tensor,adv_patch=delta,positions=image_locations, randomize=True)
                    # Add a batch axis of one for concatination
                    modified_image = modified_image[np.newaxis, :, :, :]
                    if first == True:
                        first = False
                        saved = modified_image
                    else:
                        #Perform concatination with all images in the batch
                        saved = torch.cat((saved,modified_image), axis=0)

                # Calculate detections on the batch
                pred = detect(saved,model,device)

                # For each torch.tensor in predictions..
                for k in range(len(pred)):
                    if pred[k] != []:
                        # Extract confidence
                        confidence = pred[k][0][4]
                        # Define loss function
                        conf_loss = torch.square(confidence)
                        # Backpropagate
                        conf_loss.backward()
                        # Calculate the data grad towards the function minimum
                        current_data_grad = -1*delta.grad
                        # Increment variables
                        sum_confidence = sum_confidence + confidence.item()
                        sum_grad = sum_grad+current_data_grad
                        sum_loss = sum_loss + gt_confidence[k]-confidence
                if pred == []:
                    # If no detection is returned, increment the number of generations that have finished
                    num_finished+=1
                    continue

                if num_finished == num_random_generations:
                    # If all random locations return no detections, end training
                    logging.info("All random generations have returned no detections")
                    sum_confidence = torch.tensor(0)
                    break

            # Calculate the accumulated loss, confidence, and data_grad for the epoch
            loss = (sum_loss/(num_random_generations*image_tensor.shape[0]))
            confidence = torch.tensor(sum_confidence/(num_random_generations*image_tensor.shape[0]))
            data_grad = (sum_grad /(num_random_generations*image_tensor.shape[0]))

            # Perform convex combination of current and previous gradients, ie. momentum calculation
            if last_grad is not None:
                data_grad = alpha*data_grad + (1-alpha)*last_grad
            last_grad = data_grad.clone()

            # Modify delta by lr according to the data_grad
            delta = fgsm_attack(delta, lr, data_grad)
            # Clamp delta
            delta = delta.data.clamp(0, 255)
            # Reatach a gradient to delta
            delta = Variable(delta, requires_grad=True)



        else:
            # Initialize variables
            sum_grad = 0
            sum_confidence = 0
            sum_loss = 0
            num_finished = 0
            first = True

            # For every image in the batch
            for j in range(image_tensor.shape[0]):
                # Isolate a single image tensor
                single_tensor = image_tensor[j, :, :, :]
                # Combine image and delta at given x,y
                modified_image = apply_patch(patched_img=single_tensor, adv_patch=delta, positions=image_locations,randomize=False,givenY=givenY,givenX=givenX)
                # Add a batch axis for concatination
                modified_image = modified_image[np.newaxis, :, :, :]
                if first == True:
                    first = False
                    saved = modified_image
                else:
                    # Perform concatination with all images in the batch
                    saved = torch.cat((saved, modified_image), axis=0)

            # Calculate detections on the batch
            pred = detect(saved, model, device)

            # For each torch.tensor in predictions..
            for k in range(len(pred)):
                if pred[k] != []:
                    # Extract confidence
                    confidence = pred[k][0][4]
                    # Define loss function
                    conf_loss = torch.square(confidence)
                    # Backpropagate
                    conf_loss.backward()
                    # Calculate the data grad towards the function minimum
                    current_data_grad = -1 * delta.grad
                    # Increment variables
                    sum_confidence = sum_confidence + confidence.item()
                    sum_grad = sum_grad + current_data_grad
                    sum_loss = sum_loss + gt_confidence[k] - confidence
            if pred == []:
                # If no detection is returned, increment the number of generations that have finished

                # If all  images in batch return no detections, end training
                logging.info("All random generations have returned no detections")
                best_confidence = 0
                best_epoch = t
                loss = torch.tensor(-1)
                confidence = torch.tensor(0)
                # Save the delta as a .jpg files
                cv2.imwrite(os.path.join(result_path, 'jpg_files', 'patch_{:05d}_conf={:.2f}.jpg'.format(t, best_confidence)), delta.detach().permute(1, 2, 0).numpy())
                numpy_save = delta.detach().numpy()
                # Save the delta as a .npy file
                np.save(os.path.join(result_path, 'numpy_files',
                                     'Checkpoint_{:05d}_conf={:.2f}.npy'.format(t, best_confidence)), numpy_save)
                # Create the save path for the save state
                state_save_path = os.path.join(result_path,
                                               'state_files/Checkpoint_{:05d}_conf={:.2f}.csv'.format(t, best_confidence))
                # Create the state dictionary
                save_dict = {"epoch": [t], "best_epoch": [best_epoch], "best_confidence": [best_confidence],'loss': [float(loss.item())], "confidence": [float(confidence.item())]}
                # Save the state dictionary
                saveData(save_dict, save_dir=state_save_path)
                # Log current state to console
                logging.info( "Epoch: {} of {}:   Loss: {:+.4e}  Confidence: {:.4f}  Best Confidence: {:.4f} at epoch {:d}".format(t, epochs,loss.item(),confidence.item(),best_confidence, best_epoch))
                sys.exit()

            # Calculate the accumulated loss, confidence, and data_grad for the epoch
            loss = (sum_loss / image_tensor.shape[0])
            confidence = torch.tensor(sum_confidence) / image_tensor.shape[0]
            data_grad = (sum_grad /image_tensor.shape[0])

            # Perform convex combination of current and previous gradients, ie. momentum calculation
            if last_grad is not None:
                data_grad = alpha * data_grad + (1 - alpha) * last_grad
            last_grad = data_grad.clone()

            # Modify delta by lr according to the data_grad
            delta = fgsm_attack(delta, lr, data_grad)
            # Clamp delta
            delta = delta.data.clamp(0, 255)
            # Reatach a gradient to delta
            delta = Variable(delta, requires_grad=True)


        # Log to tensorboard
        if writer is not None and return_output == False:  # this is the tensorboard logging for the training loop
            writer.add_scalar('LearningRate', lr, t)
            writer.add_scalar('Loss', loss.item(), t)
            writer.add_scalar('Confidence', confidence.item(), t)

        # If confidence is lower than previous best confidence,then save
        if confidence.item() < best_confidence:
            # Update best epoch and best confidence
            best_epoch = t
            best_confidence = confidence.item()
            checkpoint_save_delta = delta.detach().permute(1, 2, 0).numpy()

            if return_output == False:
                # Save the delta as a .jpg file
                cv2.imwrite(os.path.join(result_path,'jpg_files', 'patch_{:05d}_conf={:.2f}.jpg'.format(t, best_confidence)),checkpoint_save_delta)
                numpy_save = delta.detach().numpy()
                # Save the delta as a .npy file
                np.save(os.path.join(result_path,'numpy_files','Checkpoint_{:05d}_conf={:.2f}.npy'.format(t,best_confidence)), numpy_save)
                # Create the save path for the save state
                state_save_path = os.path.join(result_path,'state_files/Checkpoint_{:05d}_conf={:.2f}.csv'.format(t,best_confidence))
                # Create the state dictionary
                save_dict = {"epoch":[t],"best_epoch":[best_epoch],"best_confidence": [best_confidence],'loss':[float(loss.item())],"confidence":[float(confidence.item())]}
                # Save the state dictionary
                saveData(save_dict,save_dir=state_save_path)
        # Log current state to console
        logging.info("Epoch: {} of {}:   Loss: {:+.4e}  Confidence: {:.4f}  Best Confidence: {:.4f} at epoch {:d}".format(t, epochs, loss.item(), confidence.item(), best_confidence, best_epoch))

        # Check patience, potentially end training if patience has expired
        if t - best_epoch >= patience:
            logging.info('Out of patience ... early stopping @ Epoch: {}\n'.format(t))
            break

    # Once training is over, save all deltas
    if return_output == False:
        save_delta = delta.detach().permute(1, 2, 0).numpy()
        # Save delta as a .jpg
        cv2.imwrite(os.path.join(result_path, 'jpg_files', 'final_patch.jpg'), save_delta)
        numpy_save = delta.detach().numpy()
        # Save delta as a .npy
        np.save(os.path.join(result_path, 'numpy_files', 'final_model.npy'), numpy_save)
        elapsed_time = datetime.datetime.now() - start_time
        # Log final information to console
        logging.info('Elapsed Training Time: {:.4f} hrs \n\n'.format(elapsed_time.total_seconds() / 3600.0))
        logging.info("Final confidence lower than {:.4f} (saving patch one optimization step later)".format(confidence.item()))


    else: # Otherwise ... return best confidence for the training run
        return (best_confidence.item())


@gin.configurable()
def evaluate(names,model,device,image_path,numpy_patch_path,command,height,width,offset,givenX=None,givenY=None,step_size=10):

    """
    This function evaluates a loaded delta. Use command = 'all' for a grid sweep of delta, evaluating at every location, command = 'single' for a single point evalaution of delta at
    a specific (x,y) and command = 'random' for 100 random locations using the offset paramater to limit location
    Params:
       names --> The list of names associated with yoloV3's returned class label (list)
       model --> The Nnet model (darknet model)
       device --> The device to run on (either 'cpu' or 'cuda:0')
       image_path --> The path to the data directory for the images (string)
       numpy_patch_path --> The path to the saved adversarial patch (string)
       command --> The desired type of evaluation ('single','all','random')
       height --> The desired height of all images (int)
       width --> The desired width of all images (int)
       offset --> The minimum distance between the bounding box and the edge of the patch (int)
       givenX --> Optional parameter for use with command = 'single' for the desired x location of the delta (OPTIONAL: int)
       givenY --> Optional parameter for use with command = 'single' for the desired y location of the delta (OPTIONAL: int)

    Returns:
       None --> Prints results to the console

     """
    logging.info('Running evaluate')

    # Check that location is given if command = 'single'
    if command == 'single':
        assert givenY != None and givenX!= None, 'Please specify a location in config.gin when using single evaluation'

    # Load the images
    image_tensor, image_name  = load_image(image_path,height,width)

    # Get a base prediction for the unmodified images
    gt = detect(image_tensor, model, device)

    # Initialize variable
    total_confidence = 0

    # Print base prediction information
    for i in range(len(gt)):
        image_confidence = gt[i][0][4].item()
        logging.info("Detected a {} with {:5.7f} % certainty on initial image load of image {}".format(names[int(gt[i][0][5].item())],image_confidence ,i+1))
        total_confidence = total_confidence + image_confidence
    print('Average confidence on initial load of images ', total_confidence/len(gt))

    # Load delta
    delta = torch.from_numpy(np.load(numpy_patch_path))
    delta = Variable(delta, requires_grad=True)

    # Main logic flow
    if command == 'single':
        logging.info('Performing evaluate on a single location ')

        # For each image in the batch
        for k in range(image_tensor.shape[0]):
            # Torch.Tensor form of the bounding box
            truth = gt[k][0:4].detach()[0]

            # Seperate an individual image and calculate allowable patch locations for that image
            single_image = image_tensor[k, :, :, :]
            image_locations = calculate_positions(single_image, delta, truth, offset)

            # Combine image and delta at givenX and given Y
            combined_img = apply_patch(patched_img=single_image, adv_patch=delta, positions=image_locations,randomize=False,givenX=givenX,givenY=givenY)

            # Add a batch dimension for detector
            combined_img = combined_img[np.newaxis, :, :, :]

            # Calculate detections
            pred = detect(combined_img,model,device)

            # Report results
            if pred != []:
                pred = pred[0]
                print("Detected a", names[int(pred[0][5].item())], 'with', pred[0][4].item(), "% certainty while running evaluate on at location",str(givenX)+","+str(givenY),"on image", k+1)
            else:
                print("No detection returned while running evaluate on at location",str(givenX)+","+str(givenY),"on image", k+1)

    elif command == 'all':
        logging.info("Performing evaluate on all locations")

        # For every image in the batch...
        for k in range(image_tensor.shape[0]):
            # List format of the bounding box Deletion
            ground_truth = gt[k][0:4].detach().tolist()[0]
            # Torch.Tensor form of the bounding box
            truth = gt[k][0:4].detach()[0]
            # Seperate an individual image and calculate allowable patch locations for that image
            single_image = image_tensor[k, :, :, :]
            image_locations = calculate_positions(single_image, delta, truth, offset)

            # Initialize variables before grid search
            sum = 0
            count = 0

            # For every x in the width of the image
            for i in range(0,single_image.shape[2],step_size):

                # For every y in the height of the image
                for j in range(0,single_image.shape[1],step_size):

                    # If the delta will overlap with the bounding box, then ignore that calculation
                    if i + delta.shape[1] > ground_truth[0] and i < ground_truth[2]:
                        if j + delta.shape[2] > ground_truth[1] and 1 <ground_truth[3]:
                            continue
                    # Else, perform the calculation
                    else:
                        # Combine image and delta at givenX = i and givenY = j
                        combined_img = apply_patch(patched_img=single_image, adv_patch=delta, positions=image_locations, randomize=False, givenX=i, givenY=j)

                        # Add a batch dimension for the detector
                        combined_img = combined_img[np.newaxis, :, :, :]

                        # Calculate detection
                        pred = detect(combined_img, model, device)

                        # If detection is not none, increment sum and count
                        if pred != []:
                            pred = pred[0][0]
                            sum += pred[4].item()
                            count +=1

                        # Else, increment count
                        else:
                           count +=1

            # Report results
            print("Average confidence over",count,"locations = ", sum/count,"on image",k+1)

    elif command == 'random':
        logging.info("Performing evaluate on random locations")

        # For each image in the batch...
        for k in range(image_tensor.shape[0]):
            # Torch.Tensor form of the bounding box
            truth = gt[k][0:4].detach()[0]
            # Seperate an individual image and calculate allowable patch locations for that image
            single_image = image_tensor[k, :, :, :]
            image_locations = calculate_positions(single_image, delta, truth, offset)
            # Initialize variables before random calculation
            sum = 0
            count = 0

            # I have found sample size 100 to be most representative while minimizing computational time... This can be changed to anything else desired
            for i in range(100):

                # Combine image and delta at a random location
                combined_img = apply_patch(patched_img=single_image, adv_patch=delta, positions=image_locations, randomize=True)

                # Add a batch dimension for the detector
                combined_img = combined_img[np.newaxis, :, :, :]

                # Calculate detection
                pred = detect(combined_img, model, device)

                # If detection is not none, increment sum and count
                if pred != []:
                    pred = pred[0]
                    pred = pred[0]
                    sum += pred[4].item()
                    count += 1

                # Else, increment count
                else:
                    count += 1

            # Report results
            print("Average confidence over", count, "locations = ", sum / count, "on image", k)

    elif command == 'noise':
        logging.info("Performing evaluate on random locations with random noise ")

        # Initialize variables before random calculation
        noise_sum = 0
        noise_count = 0
        adv_sum = 0
        adv_count = 0
        adv_noise_sum = 0
        adv_noise_count = 0


        for k in range(image_tensor.shape[0]):
            # Torch.Tensor form of the bounding box
            truth = gt[k][0:4].detach()[0]
            # Seperate an individual image and calculate allowable patch locations for that image
            single_image = image_tensor[k, :, :, :]
            image_locations = calculate_positions(single_image, delta, truth, offset)

            # I have found sample size 100 to be most representative while minimizing computational time... This can be changed to anything else desired
            for i in range(100):
                # Randomly select position to insure patch and overlay are applied at the same position
                x0, y0 = random.sample(image_locations, 1)[0]

                blurred_image = single_image.permute(1,2,0).int().numpy()
                blur = cv.GaussianBlur(blurred_image.astype(np.uint8), (5, 5), 0)
                #plt.imshow(blur)
                blur = torch.tensor(blur).permute(2,0,1)
                blur = blur[np.newaxis,:,:,:]

                # Calculate detection
                pred = detect(blur, model, device)

                if pred != []:
                    pred = pred[0]
                    noise_sum += pred[0][4].item()
                    noise_count += 1

                # Else, increment noise_count
                else:
                    noise_count += 1


                # Combine image and delta at a random location
                combined_img = apply_patch(patched_img=single_image, adv_patch=delta, positions=image_locations,randomize=False,givenX=x0,givenY=y0)

                # Add a batch dimension for the detector
                combined_img = combined_img[np.newaxis, :, :, :]

                # Calculate detection
                pred = detect(combined_img, model, device)

                # If detection is not none, increment adv_sum and adv_count
                if pred != []:
                    pred = pred[0]
                    adv_sum += pred[0][4].item()
                    adv_count += 1

                # Else, increment adv_count
                else:
                    adv_count += 1

                # Remove batch dimension
                combined_img = combined_img[0,:,:,:]

                # Permute image for open CV Gaussian Blur operator
                blurred_image = combined_img.permute(1,2,0).int().numpy()

                # Apply blur colonel
                blur = cv.GaussianBlur(blurred_image.astype(np.uint8), (5, 5), 0)

                # Permute back and add a batch dimension for detector
                blur = torch.tensor(blur).permute(2,0,1)
                full_blur = blur[np.newaxis,:,:,:]

                # Calculate detection
                pred = detect(full_blur, model, device)

                # If detection is not none, increment adv_noise_sum and adv_noise_count
                if pred != []:
                    pred = pred[0]
                    adv_noise_sum += pred[0][4].item()
                    adv_noise_count += 1

                # Else, increment count
                else:
                    adv_noise_count += 1

            # Report results
        print("Average confidence over", noise_count, "locations with blur = ", noise_sum / noise_count,
              "on all images")
        print("Average confidence over", adv_count, "locations with adversarial patch = ", adv_sum / adv_count,
              "on all images")
        print("Average confidence over", adv_noise_count, "locations with adversarial patch and blur  = ",
              adv_noise_sum / adv_noise_count, "on all images")

        # Handle case where unknown command is given

    else:
        logging.info("Unknown command entered")


@gin.configurable()
def optimize(names,model,device,height,width,offset,learning_rates,generations_choices):
    """
    This function performs a grid search of learning rates and number of random generations to try and find the best combination.
    I WOULD NOT ADVISE USING THIS because the time complexity is terrible. Implement point swarm optimization or some other optimization strategy for better results. Doing this
    optimization will yield results, it is just very slow.
     Params:
         names --> The list of names associated with yoloV3's returned class label (list)
         model --> The Nnet model (darknet model)
         device --> The device to run on (either 'cpu' or 'cuda:0')
         height --> The desired height of all images (int)
         width --> The desired width of all images (int)
         offset --> The minimum distance between the bounding box and the edge of the patch (int)
         learning_rates --> A list of learning rates to try
         generations_choices --> A list of numbers of random generations to try

     Returns:
         None --> Prints results to the console

     """
    logging.info("Running optimze")

    # Start the timer
    optimize_start = time.perf_counter()

    # Initialize variables
    best_conf = 1
    best_learning_rate = -1
    best_number_generations = -1

    # For every learning rate in the list of learning rates
    for i in range(len(learning_rates)):
        l = learning_rates[i]

        # For every number of random generations in the list of generations choices
        for j in range(len(generations_choices)):
            g = generations_choices[j]

            # Set run name and generate a patch with the given parameters
            run_name = "Optimization_lr_"+str(i)+"gen"+str(l)
            best_confidence = generate_attack(names=names,writer=None,height=height,width=width,offset=offset,model=model,run_name=run_name,device=device,lr=l, num_random_generations=g,return_output=True)

            # Report results for that run
            print('Confidence using lr',l,"and num generations",g,"=",best_confidence)

            # If confidence is a new low, save it
            if best_confidence < best_conf:
                best_conf = best_confidence
                best_learning_rate = l
                best_number_generations = g

    # Stop the timer
    optimize_end = time.perf_counter()

    # Print results to console
    print("Best confidence acheived using lr",best_learning_rate,"and num_generations",best_number_generations,'. Final confidence was',best_conf)
    print("Optimization completed in ",optimize_end-optimize_start,"seconds")


@gin.configurable()
def visualize(names,model,device,image_path,numpy_patch_path,offset,height,width, save_plots=True):
    """
    This function visualizes bounding boxes and adversarial patches on data.
    Params:
       names --> The list of names associated with yoloV3's returned class label (list)
       model --> The Nnet model (darknet model)
       device --> The device to run on (either 'cpu' or 'cuda:0')
       image_path --> The path to the data directory for the images (string)
       numpy_patch_path --> The path to the saved adversarial patch (string)
       offset --> The minimum distance between the bounding box and the edge of the patch (int)
       height --> The desired height of all images
       width --> The desired width of all images

    Returns:
         None --> Shows images on display

    """
    logging.info("Running visualize ")

    # Load the images
    image_tensor, image_names = load_image(image_path,height,width)

    # Get the base predictions for the unmodified images
    base_prediction = detect(image_tensor, model, device)

    # For every image in the batch...
    for k in range(image_tensor.shape[0]):
        # Generate the label for the image
        string = names[int(base_prediction[0][0][5].item())] + " " + str(round(base_prediction[k][0][4].item(),4))

        ground_truth = base_prediction[k][0:4].detach().tolist()[0] # List form of the bounding box
        truth = base_prediction[k][0:4].detach()[0] # Torch.Tensor for of the bounding box

        # Seperate the batch to obtain a tensor with batch size 1
        single_image = image_tensor[k, :, :, :]

        # Perform plotting using Pyplot from Matplotlib
        fig = plt.figure(k,figsize=(6,4))
        ax = fig.add_subplot(1,3,1)
        ax.imshow(single_image.permute(1,2,0))
        ax.set_title('Original')
        rect = patches.Rectangle((ground_truth[0],ground_truth[1]),ground_truth[2]-ground_truth[0],ground_truth[3]-ground_truth[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.text(ground_truth[0],ground_truth[1],string,color='r')
        ax.add_patch(rect)


        # Load the adversarial patch
        delta = torch.from_numpy(np.load(numpy_patch_path))

        # Calculate acceptable positions foe the patch
        image_locations = calculate_positions(single_image, delta, truth, offset)

        # Combine the image and the path
        combined_img = apply_patch(patched_img=single_image,adv_patch=delta,positions=image_locations)

        # Add a batch dimension
        combined_img = combined_img[np.newaxis,:,:,:]

        # Calculate predictions for the combined image
        predictions = detect(combined_img, model, device)

        # If an object was detected ...
        if predictions != []:
            # Generate the label for the image
            string = names[int(predictions[0][0][5].item())]  + " " + str(round(predictions[0][0][4].item(), 4))
            # Calculate the new bounding box
            bounding_box = predictions[0][0:4].detach().tolist()[0]

            # Perform plotting using Pyplot from Matplotlib
            ax = fig.add_subplot(1,3,2)
            ax.imshow(combined_img[0,:,:,:].int().permute(1, 2, 0))
            rect = patches.Rectangle((bounding_box[0], bounding_box[1]), bounding_box[2] - bounding_box[0],
                                     bounding_box[3] - bounding_box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.text(bounding_box[0], bounding_box[1], string, color='b')
            ax.add_patch(rect)
            ax.set_title('Partial Attack')

        # If no object was detected...
        else:
            # Perform plotting using Pyplot from Matplotlib
            ax = fig.add_subplot(1,3,2)
            ax.imshow(combined_img[0, :, :, :].int().permute(1, 2, 0))
            ax.set_title('Lethal Attack')

        # Remove batch dimension for open CV's Gaussian Blur function
        combined_img = combined_img[0,:,:,:]

        # Permute image and convert to numpy array for open CV Gaussian Blur operator
        blurred_image = combined_img.permute(1, 2, 0).int().numpy()

        # Apply blur colonel
        blur = cv.GaussianBlur(blurred_image.astype(np.uint8), (5, 5), 0)

        # Permute back and add a batch dimension for detector
        blur = torch.tensor(blur).permute(2, 0, 1)
        full_blur = blur[np.newaxis, :, :, :]

        # Calculate detection
        predictions = detect(full_blur, model, device)

        # If an object was detected ...
        if predictions != []:
            # Generate the label for the image
            string = names[int(predictions[0][0][5].item())]  + " " + str(round(predictions[0][0][4].item(), 4))
            # Calculate the new bounding box
            bounding_box = predictions[0][0:4].detach().tolist()[0]

            # Perform plotting using Pyplot from Matplotlib
            ax = fig.add_subplot(1,3,3)
            ax.imshow(full_blur[0,:,:,:].int().permute(1, 2, 0))
            rect = patches.Rectangle((bounding_box[0], bounding_box[1]), bounding_box[2] - bounding_box[0],
                                     bounding_box[3] - bounding_box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.text(bounding_box[0], bounding_box[1], string, color='b')
            ax.add_patch(rect)
            ax.set_title('Attack with Gaussian Blur')
        plt.show()

        # Save the plots
        if save_plots:
            # Construct the figure directory within the directory where the patch is
            strs = numpy_patch_path.split('/')
            fig_dir = os.path.join(*strs[:-2], 'figures')
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            output_name = image_names[k]
            index = output_name.rfind(".")
            output_name = output_name[:index] + "_adversarial_result.png"
            fig.savefig(os.path.join(fig_dir, output_name))


@gin.configurable()
def tensorboard_logging(command, run_name, No_saved_file,save_path='./tf_logs'):
    """
    This function function generates a writer object for tensorboard logging
    Params:
        command --> Use for the logging (string)
        run name --> Run name (string)
        No_Saved_file --> If there is a saved log file (boolean)
        save_path --> location to save the tf log files (string)

    Returns:
        writer --> A tensorboard writer object for logging (object)
    """
    #Safety
    writer = None

    #Create directory structure
    if command == 'train':
        now = datetime.datetime.now()
        # Create a tensorboard logging directory with current time/date
        tf_logdir = os.path.join(save_path, run_name, now.strftime("%Y.%m.%d:%H.%M.%S"))
        if No_saved_file == False:
            # find the most recent log file
            log_files = os.listdir(os.path.join(save_path))
            log_files.sort()
            tf_logdir = os.path.join('tf_logs', log_files[-1])
    else:
        tf_logdir = None

    #Create an instance of the writer with the directory tf_logdir
    if (tf_logdir != None):
        writer = SummaryWriter(tf_logdir)

    #Return the writer
    return writer


def saveData(dict,save_dir):
    """
    This function saves any dictionary to a csv file
    Params:
       dict --> The dictionary to save (dictionary)
       save_dir --> The directory to save the dictionary to (string)

    Returns:
        None
    """
    # Create a pandas dataframe from the dictionary
    data_frame = pd.DataFrame.from_dict(dict)

    # Save the dataframe
    data_frame.to_csv(save_dir, index=False, index_label=False)


@gin.configurable()
def run_script(execute, log_file_name, run_name,height,width,offset,coco_names, device='cpu',seed = 152):
    """
       This function controls the flow of the script by calling helper functions.
       Params:
          execute --> The command for what the user wants the script to do (string)
          log_file_name --> Name of the log file (string)
          run_name --> Name of the run (string)
          height --> Desired height of the imagery (int)
          width --> Desired width of the imagery (int)
          offset --> The minimum distance between the bounding box and the edge of the patch (int)
          coco_names --> The path to the names file for yolov3 to interpret detection outputs (string)
          device --> The device to run on (either 'cpu' or 'cuda:0')
          seed --> The random seed for all random number generation (int)

       Returns:
          None --> Prints results to the console and calls other functions

        """

    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load the model
    model = Darknet('yolov3/cfg/yolov3-tiny.cfg', 512)

    # Load the state dictionary
    weights = 'yolov3/yolov3-tiny.pt'
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Set the device
    device = torch_utils.select_device(device=device if ONNX_EXPORT else '')

    # Load the list of names for use with yolov3
    names = load_classes(coco_names)

    # Create the result directory
    result_path = os.path.join('results', run_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(os.path.join(result_path, 'numpy_files'))
        os.makedirs(os.path.join(result_path, 'jpg_files'))
        os.makedirs(os.path.join(result_path, 'state_files'))

    # Setup the logger so we can log console output to the file specified in 'log_file_name'
    log_path = os.path.join(result_path, log_file_name)
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(log_path)])

    # Disable logging for matplotlib so it doesn't pollute console with font information
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.INFO)

    # Log the config file
    logging.info('')
    logging.info('================== Gin configuration start =======================')
    config_file = sys.argv[1]  # Parse config file name from option --gin_config=config_file_name
    logging.info(open(config_file, 'r').read())
    logging.info('================== Gin configuration end =======================')
    logging.info('')


    if execute == 'train': # Start training
        writer = tensorboard_logging('train', run_name, True) # Create a writer object
        generate_attack(names,writer, model, device, run_name,height=height,width=width,offset=offset)

    elif execute == 'evaluate': # Start evaluation
        evaluate(names=names,model=model,device=device,height=height,width=width,offset=offset)

    elif execute == 'visualize': # Start visualization
        visualize(names,model,device,height=height,width=width,offset=offset)

    elif execute == 'optimize': # Start optimizaiton
        optimize(names=names,height=height,width=width,offset=offset,model=model,device=device)

    # Handle unknown case
    else:
        logging.info("Unknown process requested -- try again")


if __name__ == '__main__':

    # Check that a config.gin file was specified as a command line parameter
    assert len(sys.argv) > 1, 'Missing path to config file as an argument'

    # Parse the config.gin file
    gin.parse_config_file(sys.argv[1])

    # Run the script
    run_script()

