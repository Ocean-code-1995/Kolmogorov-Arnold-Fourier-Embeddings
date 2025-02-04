# Fourier-based embedding layer
import numpy as np

import torch as th
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import repeat

class NaiveFourierKANLayer(th.nn.Module):
    """ link: https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py
    """
    def __init__( self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer,self).__init__()
        self.gridsize = gridsize
        self.addbias  = addbias
        self.inputdim = inputdim
        self.outdim   = outdim

        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high gridsizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        grid_norm_factor = (th.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)

        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance
        #independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter( th.randn(2,outdim,inputdim,gridsize) /
                                                (np.sqrt(inputdim) * grid_norm_factor ) )
        if( self.addbias ):
            self.bias  = th.nn.Parameter( th.zeros(1,outdim))

    #x.shape ( ... , indim )
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = th.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = th.reshape( th.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = th.reshape(x,(x.shape[0],1,x.shape[1],1) )
        #This should be fused to avoid materializing memory
        c = th.cos( k*xrshp )
        s = th.sin( k*xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y =  th.sum( c*self.fouriercoeffs[0:1],(-2,-1))
        y += th.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = th.reshape( y, outshape)
        return y




class PatchEmbedding(nn.Module):
    """
    PatchEmbedding converts an input image into a sequence of patches for transformer models.

    Parameters
    ----------
    img_size : int
        Size of the input image (assumes square images).
    patch_size : int
        Size of each patch (height and width).
    in_channels : int
        Number of input channels (e.g., 3 for RGB).
    embed_size : int
        Dimensionality of the patch embeddings.
    embedding_type : str, optional, default='conv'
        Type of embedding ('conv', 'linear', 'fourier').
    fourier_params : dict, optional
        Parameters for the Fourier embedding if `embedding_type='fourier'`.

    Attributes
    ----------
    proj : nn.Sequential
        Layer for projecting input images to patch embeddings.
    cls_token : nn.Parameter
        Learnable token prepended to the patch sequence.
    positions : nn.Parameter
        Positional embeddings added to the sequence.

    Methods
    -------
    forward(x)
        Transforms `x` into a sequence of patch embeddings with positional encodings.

    Notes
    -----
    - The `Rearrange` operations reshape the input tensor to match the format required for different embedding types.
    - 'conv': [batch_size, embed_size, h, w] -> [batch_size, num_patches, embed_size].
    - 'linear', 'fourier': [batch_size, in_channels, h*p1, w*p2] -> [batch_size, num_patches, patch_vector_size].
    """
    def __init__(self, img_size, patch_size, in_channels, embed_size, embedding_type='conv', fourier_params=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.input_dim = patch_size * patch_size * in_channels

        if embedding_type == 'conv':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e h w -> b (h w) e')  # Explicitly define h and w here
            )
        elif embedding_type == 'linear':
            self.proj = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(self.input_dim, embed_size)
            )
        elif embedding_type == 'fourier':
            if fourier_params is None:
                fourier_params = {}

            self.proj = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                NaiveFourierKANLayer(self.input_dim, embed_size, **fourier_params)
            )
        else:
            raise ValueError("Invalid embedding_type. Choose 'conv', 'linear', or 'fourier'")

        self.cls_token = nn.Parameter(th.randn(1, 1, embed_size))
        self.positions = nn.Parameter(th.randn((img_size // patch_size) ** 2 + 1, embed_size))

    def forward(self, x):
        """
        Forward pass: Transforms the input image into a sequence of patch embeddings with positional encodings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_channels, img_size, img_size].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, num_patches + 1, embed_size], where `num_patches` is the number of
            patches in the image, and the additional dimension corresponds to the classification token.

        Notes
        -----
        - `cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)`:
            This line replicates the learnable classification token (`cls_token`) across the batch dimension `b`.
            The `repeat` function is used to create a tensor of shape [batch_size, 1, embed_size] by repeating 
            the `cls_token` for each image in the batch, preparing it to be concatenated with the patch embeddings.
        """
        b, _, _, _ = x.shape
        x = self.proj(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = th.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x