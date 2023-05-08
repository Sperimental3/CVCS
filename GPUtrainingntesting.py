import torch
import Embedder
import TripletsBuilder as ds
from torch.nn import TripletMarginLoss
from sklearn.neighbors import KNeighborsClassifier
# import random
from sklearn.metrics import accuracy_score


def create_triplet_miniBatch(dim, tensor, iter_num):

    miniBatch = torch.empty(dim)
    # indexes = list(range(0, tensor.shape[0], 3))

    for j in range(0, dim[0], 3):
        index = iter_num * dim[0] + j

        miniBatch[j] = tensor[index]
        miniBatch[j+1] = tensor[index+1]
        miniBatch[j+2] = tensor[index+2]

        # indexes.remove(index)

    return miniBatch


def trainCNN():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print(device)

    # try_tensor = torch.randn((9, 3, 100, 100), requires_grad=True)
    # try_tensor = torch.randint(low=0, high=256, size=(9, 3, 100, 100)).type(torch.float32)
    # try_tensor.requires_grad = True
    # print(try_tensor)

    print("Building the triplets for the training of the model...")

    try_tensor, _, _, _ = ds.inputTensor()

    # print(try_tensor.shape)

    # try_tensor = try_tensor.to(device)

    # print(try_tensor.is_cuda)

    model = Embedder.Embedder(128)

    model = model.to(device)

    # the correct number of the epochs depends on the learning rate, we found 9600 to be a reasonable value
    epochs = 9600
    learning_rate = 0.001

    iter_num = 0
    # training part
    for i in range(epochs):
        # we choose a dimension of 30 for miniBatches, so 90 elements

        if (iter_num * 90) >= try_tensor.shape[0]:
            iter_num = 0
            mini_batch = create_triplet_miniBatch((90, 3, 100, 100), try_tensor, iter_num)
            iter_num += 1
        else:
            mini_batch = create_triplet_miniBatch((90, 3, 100, 100), try_tensor, iter_num)
            iter_num += 1

        mini_batch = mini_batch.to(device)

        # print(mini_batch.is_cuda)

        output = model(mini_batch)
        # print(mini_batch.shape)

        anchors = output[0::3]
        positives = output[1::3]
        negatives = output[2::3]

        # print(mini_batch.device, output.device)

        # print(anchors.shape, anchors.device)
        # print(positives.shape, positives.device)
        # print(negatives.shape, negatives.device)

        triplet_loss = TripletMarginLoss(margin=0.1)    # reduction="none" for evaluating the margin

        loss = triplet_loss(anchors, positives, negatives)

        # print(loss)

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        model.zero_grad()

        del mini_batch
        del output
        del anchors
        del positives
        del negatives

        torch.cuda.empty_cache()

        print(f"Epoch number {i} is finished, the loss is: {loss}")

    return model


def do_everything():

    # the average accuracy score over a 5 loop cycle of testing and training of the CNN with always new 10800 triplets
    # is: 0.8920353651046753, nice(we think), this was without edges.

    number_of_loops = 5
    accuracy_tensor = torch.empty(number_of_loops)

    for i in range(number_of_loops):
        print("Training of the CNN...")
        model = trainCNN()
        print("Training of the CNN is finished.")

        # model = model.to("cpu")

        print("Building the triplets for the testing part...")

        X_train, y_train, X_test, y_test = ds.inputTensor()

        # print(X_train.shape, len(y_train))
        # print(X_test.shape, len(y_test))

        print("Calculating the embedding vectors for train and test of the KNN model...")

        output = torch.empty((X_train.shape[0], 128))
        # output2 = torch.empty((X_test.shape[0], 128))
        
        # torch.cuda.empty_cache()
        
        # forward part with cuda
        for j in range(X_train.shape[0]//90):
            with torch.no_grad():
                mini_batch = create_triplet_miniBatch((90, 3, 100, 100), X_train, j)

                mini_batch = mini_batch.to("cuda:0")

                # print(mini_batch.is_cuda)

                output[j*90:j*90 + 90] = model(mini_batch)
                # print(mini_batch.shape)

            # del mini_batch
            # torch.cuda.empty_cache()
    
        X_train = output

        with torch.no_grad():

            X_test = X_test.to("cuda:0")

            X_test = model(X_test)

        # del mini_batch
        # torch.cuda.empty_cache()

        X_test = X_test.to("cpu")

        # code for training on cpu
        '''
        print("Calculating the embedding vectors for train and test of the KNN model...")

        X_train = model(X_train)
        X_test = model(X_test)

        # print(X_train.shape, len(y_train))
        # print(X_test.shape, len(y_test))

        X_train = X_train.detach()
        X_test = X_test.detach()

        print("Training and testing of the KNN model...")
        '''

        # print(X_train.shape, len(y_train))
        # print(X_test.shape, len(y_test))

        modelClassifier = KNeighborsClassifier()
        modelClassifier.fit(X_train, y_train)

        y_pred = modelClassifier.predict(X_test)

        accuracy_tensor[i] = accuracy_score(y_test, y_pred)

    print(f"Our fantastic accuracy score is: {torch.mean(accuracy_tensor)}, and min: {torch.min(accuracy_tensor)}"
          f"and max: {torch.max(accuracy_tensor)}.")


