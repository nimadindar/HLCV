import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod # To be implemented by child classes.
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()
    
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################
        result = [ret_str]
        total_params = 0
        for name, module in self.named_modules():
            if len(list(module.children())) > 0:
                continue
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                result.append(f"Layer: {name or 'unnamed'}, Type: {type(module).__name__}, Trainable Parameters: {params:,}")
                total_params += params
        result.append(f"Total Trianable Parameters: {total_params:,}")
        return "\n".join(result)
    
    