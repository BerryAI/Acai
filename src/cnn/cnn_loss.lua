----------------------------------------------------------------------
-- loss functions:
--   + negative-log likelihood, using log-normalized output units (SoftMax)
--   + mean-square error
--   * more loss function could be added in this file
--   
-- I: nil
-- O: nil
-- By Chrishu
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

function cnn_loss()
	print '==> define loss function'
	if para.loss == 'nll' then	
		-- This loss requires properly normalized log-probabilities
		model:add(nn.LogSoftMax())
		criterion = nn.ClassNLLCriterion()	
	elseif para.loss == 'mse' then
		-- for MSE, tanh is needed to restrict the model's output
		model:add(nn.Tanh())
		criterion = nn.MSECriterion()
		criterion.sizeAverage = false
	else	
		error('unknown -loss')		
	end
end