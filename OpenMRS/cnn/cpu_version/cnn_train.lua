----------------------------------------------------------------------
-- This file define the training function of CNN
--
-- I: nil
-- O: nil
-- By Chrishu
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- use to display progress bars, not necessary
require 'optim'   -- an optimization package, for online and batch methods

function optzset()
	print '==> configuring optimizer parameters'
	
	if para.optimization == 'LBFGS' then
	   optimState = {
	      learningRate = para.learningRate,
	      maxIter = para.maxIter,
	      nCorrection = 10
	   }
	   optimMethod = optim.lbfgs
	
	elseif para.optimization == 'SGD' then
	   optimState = {
	      learningRate = para.learningRate,
	      weightDecay = para.weightDecay,
	      momentum = para.momentum,
	      learningRateDecay = 1e-7
	   }
	   optimMethod = optim.sgd
	
	elseif para.optimization == 'ASGD' then
	   optimState = {
	      eta0 = para.learningRate,
	      t0 = para.trainNum * para.startAveraging
	   }
	   optimMethod = optim.asgd
	
	else
	   error('unknown optimization method')
	end
end

function train()
	print '==> defining training procedure'
	-- epoch tracker
	epoch = epoch or 1

	-- local vars
	local time = sys.clock()

	-- set model to training mode
	model:training()

	-- shuffle at each epoch
	shuffle = torch.randperm(para.trainNum)
	
	-- Retrieve parameters and gradients:
	if model then
	   parameters,gradParameters = model:getParameters()
	end

	-- do one epoch, same as torch tutorial
	print('==> doing epoch on training data:')
	print('==> online epoch # ' .. epoch .. ' [batchSize = ' .. para.batchSize .. ']')
	for t = 1,para.trainNum,para.batchSize do
		-- disp progress
		xlua.progress(t, para.trainNum)

		-- create mini batch
		local inputs = {}
		local targets = {}
		for i = t,math.min(t+para.batchSize-1,para.trainNum) do
			-- load new sample
			local input = trainData[shuffle[i]]
			local target = trainLabel[shuffle[i]]

			table.insert(inputs, input)
			table.insert(targets, target)
		end

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
				-- get new parameters
				if x ~= parameters then
					parameters:copy(x)
				end
	
				-- reset gradients
				gradParameters:zero()
	
				-- f is the average of all criterions
				local f = 0
	
				-- evaluate function for complete mini batch
				for i = 1,#inputs do
					-- estimate f
					local output = model:forward(inputs[i])
					local err = criterion:forward(output, targets[i])
					f = f + err
	
					-- estimate df/dW
					local df_do = criterion:backward(output, targets[i])
					model:backward(inputs[i], df_do)
	
					-- update confusion
					confusion:add(output, targets[i])
				end
	
				-- normalize gradients and f(X)
				gradParameters:div(#inputs)
				f = f/#inputs
	
				-- return f and df/dX
				return f,gradParameters
			end

		-- optimize on current mini-batch
		if optimMethod == optim.asgd then
			_,_,average = optimMethod(feval, parameters, optimState)
		else
			optimMethod(feval, parameters, optimState)
		end
	end

	-- time taken
	time = sys.clock() - time
	time = time / para.trainNum
	print('\n==> time to learn 1 sample = ' .. (time*1000) .. 'ms\n')

	-- next epoch
	confusion:zero()
	epoch = epoch + 1
end