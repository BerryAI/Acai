----------------------------------------------------------------------
-- This file defines the model of CNN
--
-- I: nil
-- O: nil
-- By Chrishu
----------------------------------------------------------------------

require 'nn'
require 'cunn'
require 'dp'

function cnn_model()
	model = nn.Sequential() 
	-- convolution layers 1
	model:add(nn.SpatialConvolution(1, 32, 4, 128))
	model:add(nn.ReLU())
	-- pooling layer 1
	model:add(nn.SpatialMaxPooling(4, 1, 4, 1))
	-- convolution layers 2
	model:add(nn.SpatialConvolution(32, 64, 4, 1))
	model:add(nn.ReLU())
	-- pooling layer 2
	model:add(nn.SpatialMaxPooling(2, 1, 2, 1))
	-- convolution layers 3
	model:add(nn.SpatialConvolution(64, 64, 4, 1))
	model:add(nn.ReLU())
	-- pooling layer 3
	model:add(nn.SpatialMaxPooling(4, 1, 4, 1))
	-- convolution layers 4
	model:add(nn.SpatialConvolution(64, 64, 4, 1))
	model:add(nn.ReLU())
	-- fully connected layers
	outsize = model:outside{3,1,128,999} -- output size of convolutions
	model:add(nn.Collapse(3))
	model:add(nn.Linear(outsize[2]*outsize[3]*outsize[4], 128))
	model:add(nn.Linear(128,32))
	model:add(nn.Linear(32,5))
end
