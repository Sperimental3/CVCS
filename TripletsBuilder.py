import ImagesLabeling
import HoughTransform
import ImageManipulation
import TripletLossDataset
import HoughTransformEdges
import torch
import matplotlib.pyplot as plt


# this the flag that allow us to decide if or if not to use the Canny edge detector after the HT
try_edges = 0


def inputTensor():
    # Download images from gallerie_estensi
    # ImagesScraping1.download_images()
    # Rename images and labeling
    dict_label, labels = ImagesLabeling.rename_images()

    if try_edges == 1:
        HoughTransformEdges.hough_transform()
    else:
        # Hough transform and roi pooling application
        HoughTransform.hough_transform()

    # print(HoughTransformEdges.images_batch_edges.shape)
    # HoughTransform.plot_examples()

    # print('images_batch')
    # print(HoughTransform.images_batch)
    # print(HoughTransform.images_batch.shape)
    if try_edges == 1:
        result_tensor, labels, test_tensor, test_labels = ImageManipulation.image_manipulation(
            HoughTransformEdges.images_batch_edges, dict_label, labels)
    else:
        # Application filer: rotation 90°,180°,270°,360°
        result_tensor, labels, test_tensor, test_labels = ImageManipulation.image_manipulation(
            HoughTransform.images_batch, dict_label, labels)

    # print(result_tensor.shape)
    # print(test_tensor.shape)

    # print(f"classes :{labels} and len of classes:{len(labels)}")
    # print(f"dict labels:{dict_label}")

    # here you can select the number of triplets that you desire,
    # we have find out that the best is 10800
    output_tensor, y_vector = TripletLossDataset.triplet_loss_dataset(10800, result_tensor, dict_label, labels)
    # print(f"output_tensor_shape {output_tensor.shape}")
    # print(f"len y_vector {len(y_vector)}")
    # print (output_tensor)
    # print(y_vector)

    '''
    im = torch.moveaxis(output_tensor[6], 0, 2).type(torch.int32)
    plt.imshow(im)
    plt.show()
    im = torch.moveaxis(output_tensor[7], 0, 2).type(torch.int32)
    plt.imshow(im)
    plt.show()
    im = torch.moveaxis(output_tensor[8], 0, 2).type(torch.int32)
    plt.imshow(im)
    plt.show()
    '''

    return output_tensor, y_vector, test_tensor, test_labels


