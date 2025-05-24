import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        # super(ConvNet, self).__init__()
        super().__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        in_channels = self.input_size
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for i, out_channels in enumerate(self.hidden_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if self.norm_layer is not None:
                layers.append(self.norm_layer(out_channels))
            layers.append(self.activation)
            if i < (len(self.hidden_layers) - 1):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if self.drop_prob > 0:
                layers.append(nn.Dropout(p=self.drop_prob))
            in_channels = out_channels

        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))

        self.model = nn.Sequential(*layers)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        first_conv = self.model[0]
        weights = first_conv.weight.detach().cpu().numpy()

        num_filters = weights.shape[0]
        grid_rows, grid_cols = 16, 8
        filter_size = 3
        padding = 1

        large_img_height = grid_rows * (filter_size + padding) - padding
        large_img_width = grid_cols * (filter_size + padding) - padding
        large_img = np.zeros((large_img_height, large_img_width, 3), dtype=np.float32)

        for i in range(num_filters):
            row = i//grid_cols
            col = i % grid_cols

            filter_img = weights[i]
            filter_img = self._normalize(filter_img.transpose(1,2,0))

            row_start = row * (filter_size + padding)
            row_end = row_start + filter_size

            col_start = col * (filter_size + padding)
            col_end = col_start + filter_size

            large_img[row_start:row_end, col_start:col_end, :] = filter_img
        
        plt.figure(figsize=(10,10))
        plt.imshow(large_img)
        plt.axis('off')
        plt.title('Figure of first convolutional layer filter')
        plt.show()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        

        out = self.model(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
