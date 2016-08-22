----------------------------------------------------------------------
-- Example for CNN training/testing usage
-- 
-- By Chrishu
----------------------------------------------------------------------
require 'torch'
require 'cunn'
require 'os'
require 'cnn_model.lua'
require 'cnn_loss.lua'
require 'cnn_train.lua'
require 'cnn_test.lua'
local matio = require 'matio'

-- 
print '==> processing options'

-- disable randomization, delete following 2 lines to enable
seed = 1 
torch.manualSeed(seed)

-- setting parameters
para = {datadir = 'data/'
		savedir = 'log/',
		optimization = 'SGD', -- stochastic gradient descent
		loss = 'mse', -- mean square error
		trainNum = 400, -- number os training samples
		testNum = 150, -- number os testing samples
		maxIter = 50, -- max training iteration
		learningRate = 0.05 ,
		weightDecay = 0,
		startAveraging = 1,
		momentum = 0,
		batchSize = 25, -- number of sample used in each batch (batch mode)
		noutputs = 5} -- number of output nodes

-- load a single array from file
local rawData = matio.load(para.datadir..'cnn_features.mat', 'cnn_features')
local rawLabel = matio.load(para.datadir..'cnn_labels.mat', 'cnn_labels')
trainData = (rawData:sub(1,para.trainNum)):clone()
testData = (rawData:sub(para.trainNum+1,para.trainNum+para.testNum)):clone()
trainLabel = (rawLabel:sub(1,para.trainNum)):clone()
testLabel = (rawLabel:sub(para.trainNum+1,para.trainNum+para.testNum)):clone()

-- define model
print '==> defining the model'
cnn_model()

-- define loss function
cnn_loss()
print '==> here is the loss function:'
print(criterion)

-- configuring optimizer parameters
optzset()

-- training & testing
rLoss = torch.DoubleTensor(para.maxIter)
print '==> training!'

for i = 1,para.maxIter do
	-- training procedure
	train()
	-- test procedure
	rLoss[i] = test()
end

-- generate testing output
print '==> Calculating final output!'
realOutput = model:forward(testData)

-- save training result and logger
tlabel = os.date("%y%m%d%H%M%S")
print ('==> Saving files!' .. tlabel)
torch.save('log/cnn'..tlabel..'.dat',model)
matio.save('log/output'..tlabel..'.mat',realOutput)
matio.save('log/loss'..tlabel..'.mat',rLoss)
