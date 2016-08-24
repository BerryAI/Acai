----------------------------------------------------------------------
-- This function defines the testing procedure
-- and the log procedure
--
-- I: nil
-- O: nil
-- By Chrishu
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- use to display progress bars, not necessary
require 'optim'   -- an optimization package, for online and batch methods

-- test function
function test()
	-- local vars
	local time = sys.clock()

	-- averaged param use?
	if para.average then
		cachedparams = parameters:clone()
		parameters:copy(para.average)
	end

	-- set model to evaluate mode
	model:evaluate()

	-- test over test data
	local lossAvg = 0
	print('==> testing on test set:')
	for t = 1,para.testNum do
		-- disp progress
		xlua.progress(t, para.testNum)

		-- get new sample
		local input = testData[t]

		local target = testLabel[t]

		-- test sample
		local pred = model:forward(input)
		local lossDif = pred-target
		lossAvg = lossAvg+torch.dot(lossDif,lossDif)
	end

	-- timing
	time = sys.clock() - time
	time = time / para.testNum
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

	-- print loss
	lossAvg = (lossAvg/para.testNum)/para.noutputs
	print("\n==> Loss = " .. lossAvg)

	-- averaged param use?
	if para.average then
		-- restore parameters
		parameters:copy(cachedparams)
	end
   
	-- next iteration:
	return lossAvg
end

function traintest()
 	-- local vars
	local time = sys.clock()

	-- averaged param use?
	if para.average then
		cachedparams = parameters:clone()
		parameters:copy(para.average)
	end

	-- set model to evaluate mode
	model:evaluate()

	-- test over test data
	local lossAvg = 0
	print('==> testing on train set:')
	for t = 1,para.testNum do
		-- disp progress
		xlua.progress(t, para.testNum)

		-- get new sample
		local input = trainData[t]

		local target = trainLabel[t]

		-- test sample
		local pred = model:forward(input)
		local lossDif = pred-target
		lossAvg = lossAvg+torch.dot(lossDif,lossDif)
	end

	-- timing
	time = sys.clock() - time
	time = time / para.testNum
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

	lossAvg = (lossAvg/para.testNum)/para.noutputs
	print("\n==> Loss = " .. lossAvg)

	-- averaged param use?
	if para.average then
		-- restore parameters
		parameters:copy(cachedparams)
	end
	return lossAvg
end