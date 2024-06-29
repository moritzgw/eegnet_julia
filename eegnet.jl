using Flux
using MLUtils
using Random
try
    using CUDA
catch
    println("CUDA not installed or failed to load.")
end

## function to compute number of features as input to dense output layer
function Ndense(nsamples, F1)

    return ((nsamples+1)÷4+1)÷8 * 2 * F1

end


## definition of EEGNet
function EEGNet(nb_classes, Chans, Samples; dropoutRate = 0.25, kernLength = 64, F1 = 8)

    F2 = F1 * 2
    M = convert(Int, round(Samples / 4 / 8) * F2)

    return Chain(
        x -> ndims(x) == 3 ? reshape(x, size(x,1), size(x,2), size(x,3), 1) : x,
        # Layer 1: Conv2D -> Batchnorm -> DepthwiseConv2D -> Batchnorm -> Activation -> Average Pooling -> Dropout
        # Compared with paper: norm constraint on spatial filter missing in my implementation
        Conv((1, kernLength), 1=>F1, identity; pad = (0,kernLength÷2), bias = false),
        BatchNorm(F1),
        DepthwiseConv((Chans, 1), F1=>F2; pad = (0, 0), bias = false),
        BatchNorm(F2, elu),
        x -> MeanPool((1, 4))(x),
        Dropout(dropoutRate),
        
        # Layer 2: SeparableConv2D -> Batchnorm -> Activation -> Average Pooling -> Dropout
        DepthwiseConv((1, 16), F2=>F2; pad = (0, 8), bias = false),
        Conv((1, 1), F2=>F2),  # This replaces PointwiseConv
        BatchNorm(F2, elu),
        x -> MeanPool((1, 8))(x),
        Dropout(dropoutRate),
        
        # Classification Layer
        x -> flatten(x),
        Dense(Ndense(Samples, F1), nb_classes),
        softmax
    )

end


## model struct
"""
Implements EEGNet [DOI 10.1088/1741-2552/aace8c] using Flux.

Usage:

model = eegnet(classes::Int, channels::Int, timepoints::Int; dropoutrate::Float64=0.25, kernLength::Int=64, F1::Int=8)

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

train!(model, data, labels; data_test = [], labels_test, testloss = [])

predict(model, data)

with data of dimensionality [time points, channels, samples]. When data_test, labels_test, and a testloss(ypred, y) function are provided, the field loss_test contains the loss on the test set for each epoch.

"""
mutable struct eegnet

	# model parameters
	classes::Int
	channels::Int
	timepoints::Int
	dropoutrate::Float64
	kernLength::Int
	F1::Int
	gpu::Bool
	batchsize::Int
	epochs::Int
	loss_train
	loss_test
	model

	function eegnet(classes::Int, channels::Int, timepoints::Int; dropoutrate::Float64=0.25, kernLength::Int=64, F1::Int=8)
		z = new(classes, channels, timepoints);
		z.classes = classes;
		z.channels = channels;
		z.timepoints = timepoints;
		z.dropoutrate = dropoutrate;
		z.kernLength = kernLength;
		z.F1 = F1;
		z.batchsize = 200;
		z.epochs = 100;
		z.gpu = true;
		z.loss_train = [];
		z.loss_test = [];
		z.model = EEGNet(classes, channels, timepoints; dropoutRate = z.dropoutrate, kernLength = z.kernLength, F1 = z.F1);
		return z

	end

end

## train

function train!(model::eegnet, data, labels; data_test = [], labels_test = [], testloss = [])

	T, C, N = size(data)
	data = permutedims(reshape(data, (T, C, N, 1)), [2, 1, 4, 3]) # reshape data for input
	data = Float32.(data)

	labelset = unique(labels)
	yhot = Flux.onehotbatch(labels, labelset)

	# set up training
	loss(m, x, y) = Flux.Losses.crossentropy(m(x), y)
	optimizer = Flux.setup(Flux.Optimise.Adam(), model.model)

	# shift to GPU?
	if model.gpu == true
		model.model = model.model |> gpu
		data = data |> gpu
		yhot = Float32.(yhot)	
		yhot = yhot |> gpu
		optimizer = optimizer |> gpu

	end

	if data_test != []

		TT, CT, NT = size(data_test)
		data_test = permutedims(reshape(data_test, (TT, CT, NT, 1)), [2, 1, 4, 3]) # reshape data_test for input

		if model.gpu == true

			data_test = data_test |> gpu
			data_test = Float32.(data_test)

		end

	end

	# run training

	for n in 1:model.epochs

		println("Epoch ", n, "/", model.epochs)
		
		ibatch = randperm(N)
		ibatch = ibatch[1:model.batchsize]

		xtrain = data[:, :, :, ibatch]
		ytrain = yhot[:, ibatch]
		data_train = [(xtrain, ytrain)]

		Flux.train!(loss, model.model, data_train, optimizer)

		push!(model.loss_train, loss(model.model, xtrain, ytrain))

		if data_test != []

			ytestpred = model.model(data_test)
			epochtestloss = testloss(ytestpred, labels_test)
			push!(model.loss_test, epochtestloss)

		end

	end

end

## predict
function predict(model::eegnet, data)

	T, C, N = size(data)
	data = permutedims(reshape(data, (T, C, N, 1)), [2, 1, 4, 3]) # reshape data for input
	data = Float32.(data)

	if model.gpu == true
		
		data = data |> gpu
	
	end

	y_pred = model.model(data)

	return(y_pred)

end
