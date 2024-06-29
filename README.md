# eegnet_julia
Implementation of [EEGNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c) for Julia using Flux and CUDA.

Usage:

`model = eegnet(classes::Int, channels::Int, timepoints::Int; dropoutrate::Float64=0.25, kernLength::Int=64, F1::Int=8)`

Fields:

	classes::Int
	channels::Int
	timepoints::Int
	dropoutrate::Float64
	kernLength::Int
	F1::Int
	gpu::Bool [default: true]
	batchsize::Int [default: 200]
	epochs::Int [default: 100]
	loss_train
	loss_test
	model

To train a model and make predictions run

`train!(model, data, labels; testdata = [], testlabels, testloss = [])`

`predict(model, data)`

with data of dimensionality [time points, channels, samples]. When testdata, testlabels, and a testloss function, e.g., `(yh, y) -> mean(([argmax(yh[:,n])[1] for n in 1:size(yh)[2]].-1) .== y)`, are provided, the field loss_test contains the loss on the test set for each epoch.
