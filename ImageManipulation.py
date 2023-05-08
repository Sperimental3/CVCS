# Code to apply operations on all the images
# present in a folder one by one
# operations such as rotating, cropping, 

import torch


def tensor_rotation(image_batch):
    # print(image_batch.shape)
    rot90_imgs = torch.rot90(image_batch, 1, [2, 3])
    result_tensor = torch.cat([image_batch, rot90_imgs], dim=0)
    # print(f"result_tensor: {result_tensor.shape}")
    rot180_imgs = torch.rot90(image_batch, 2, [2, 3])
    result_tensor = torch.cat([result_tensor, rot180_imgs], dim=0)
    # print(f"result_tensor: {result_tensor.shape}")
    rot270_imgs = torch.rot90(image_batch, 3, [2, 3])
    result_tensor = torch.cat([result_tensor, rot270_imgs], dim=0)
    # print(f"result_tensor: {result_tensor.shape}")

    # example function to plt rotated images (only one kind)
    """
    for img in image_batch:
        #im= torch.moveaxis(img, 0, 2).type(torch.int32)
        #print(f"im.shape {im.shape} (after torch.moveaxis(img,0,2)")
        result_tensor = img
        for i in range(4):
            rot_img = torch.rot90(img, i+1, [1, 2])
            im = torch.moveaxis(rot_img, 0, 2).type(torch.int32)
            print(f"im.shape {im.shape} (after torch.moveaxis(img,0,2)")
            # concatenate tensors
            result_tensor = torch.cat([result_tensor, rot_img], dim=0)
            print(result_tensor.shape)
            plt.imshow(im)
            plt.show()
        plt.show()
        break
    """
    return result_tensor


def tensor_linear_operator(rot_tensor, s, k):

    """
    for img in rot_tensor:
        print(img)
        im = torch.moveaxis(img, 0, 2).type(torch.int32)
        plt.imshow(im)
        plt.show()
        out = torch.round((img * s) + k).clip(0,255).type(torch.uint8)
        print(out.shape)
        print(out)
        im_out = torch.moveaxis(out, 0, 2).type(torch.int32)
        plt.imshow(im_out)
        plt.show()
        break
    plt.show()
    """

    return torch.round((rot_tensor * s) + k).clip(0, 255)


# Driver Function
# image_batch is a tensor with shape (113, 3, 100, 100)
def image_manipulation(image_batch, dict_label, labels):

    rot_tensor = tensor_rotation(image_batch)

    labels = labels + labels + labels + labels
    # print(len(labels))
    tmp_tensor = tensor_linear_operator(rot_tensor, 1.5, -50)
    result_tensor = torch.cat([rot_tensor, tmp_tensor], dim=0)
    # print(f"result_tensor.shape {result_tensor.shape}")
    tmp_tensor = tensor_linear_operator(rot_tensor, 0.7, 20)
    result_tensor = torch.cat([result_tensor, tmp_tensor], dim=0)
    # print(f"result_tensor.shape {result_tensor.shape}")
    labels = labels + labels + labels
    # print(len(labels))

    # now it's time for test dataset creation, we want to do a 90-10(more or less, to be precise is 92.7-8.3) split,
    # so we need to take just only one element(for each one of the 113 starting images) from the result_tensor
    # of 1356 elements, we choose the last images because it is the most representative.

    test_tensor = result_tensor[-113:]
    test_labels = labels[-113:]

    result_tensor = result_tensor[:-113]
    labels = labels[:-113]

    return result_tensor, labels, test_tensor, test_labels


