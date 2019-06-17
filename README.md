# Siamese and Resnet features for One shot Recognition model
This project aims to create a one shot recognition model using the features
extracted by a Siamese network and a Resnet.

It starts from the belief in which, both models does not extract the same
features of a single image; so we can use both vectors for adding extra information
to the classification task.


# Model Structure

[Model Structure](images/siaresmodel.png)


Talking about a training scenario, the model trained is showed in the image posted
above. Firstly we are going to use a Resnet pretrained model on ImageNet, so all we
need from it is the features this network is able to extract by himself. As second
model is used a Siamese network with the following inner configuration:
    
[Inner model of Siamese Network](images/siainner.png)
REF:: *link*

As this model has to be trained, we are going to set a DoubleLoss for this model.
The loss will be an addition between a BCELoss from Siamese model and a loss of 
the classification task done by the knn after processing the 2 feature vectors.