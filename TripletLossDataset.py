import torch
import random


# this function now wants the number of triplets, so the shape[0] of the final tensor will be n_triplet * 3
def triplet_loss_dataset(n_triplets, img_tensor, dict_label, labels):
    # print(f"n_triplets: {n_triplet}")
    # print(f"n_elements: {3*n_triplet} ")
    # print(f"output_tensor {output_tensor}")
    output_tensor = torch.empty((n_triplets * 3, 3, 100, 100))

    y_vector = []

    first_113_labels = list(dict_label[labels[j]] for j in range(113))
    # print(first_113_labels)
    '''
    test_list = list(range(0, img_tensor.shape[0]))
    print(len(test_list))
    '''
    for i in range(n_triplets):
        list_for_positive = list(range(0, 11))
        list_for_negative = list(range(0, 113))

        class_index = random.randrange(0, 113)
        not_the_same_index = random.randrange(0, 11)

        anchor_index = 113 * not_the_same_index + class_index

        indices_to_remove = [index for index, element in enumerate(first_113_labels) if element == first_113_labels[class_index]]
        list_for_negative = [ele for ele in list_for_negative if ele not in indices_to_remove]

        list_for_positive.remove(not_the_same_index)

        positive_index = 113 * random.choice(list_for_positive) + class_index
        negative_index = 113 * random.randrange(0, 11) + random.choice(list_for_negative)

        output_tensor[3 * i + 0] = img_tensor[anchor_index]
        y_vector.append(labels[anchor_index])

        output_tensor[3 * i + 1] = img_tensor[positive_index]
        y_vector.append(labels[positive_index])

        output_tensor[3 * i + 2] = img_tensor[negative_index]
        y_vector.append(labels[negative_index])

    # print(len(test_list))

    # print(f"output_tensor_shape {output_tensor.shape}, list len: {len(y_vector)},
    # list first elements: {y_vector[:30]}")

    return output_tensor, y_vector


