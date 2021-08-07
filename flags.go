package main

import (
	"flag"
	"fmt"
	"neural-net-go/network"
	"strconv"
	"strings"
)

type runConfig struct {
	Action           string
	ModelFile        string
	DataSetFile      string
	Epochs           int
	TestLogBatch     int
	TrainLogBatch    int
	TestParseRecord  parseRecordFunc
	TrainParseRecord parseRecordFunc
	networkConfig
}

type networkConfig struct {
	Activation        network.ActivationType
	LearningRate      float64
	RandomSeed        uint64
	InputCount        int
	OutputCount       int
	HiddenLayerCounts []int
}

func parseCmdFlags() (runConfig, error) {
	action := flag.String("action", "", "Action 'train' or 'test' against the dataset.")
	model := flag.String("model", "models/mnist.default.model", "File path of network model to load and save. If it doesn't exist a new network will be created.")
	dataset := flag.String("dataset", "", "File path of source dataset. (default \"datasets/mnist_*.csv\" where * is the action chosen)")
	epochs := flag.Int("epochs", 1, "Number of training epochs; ignored if not training.")
	activationVal := flag.String("activation", "sigmoid", "Activation function 'sigmoid' or 'tanh'.")
	learningRate := flag.Float64("learning-rate", 0.1, "Network learning rate.")
	randomSeed := flag.Uint64("random-seed", 0, "Seed for random weight generation.")
	hiddenLayerCountsStr := flag.String("hidden-layer-counts", "100", "Comma-separated list of neuron counts for hidden layers.")
	flag.Parse()
	if *dataset == "" {
		switch *action {
		case "train":
			*dataset = "datasets/mnist_train.csv"
		case "test":
			*dataset = "datasets/mnist_test.csv"
		default:
			flag.PrintDefaults()
			return runConfig{}, fmt.Errorf("unknown action '%s'", *action)
		}
	}
	var activation network.ActivationType
	switch *activationVal {
	case "sigmoid":
		activation = network.ActivationTypeSigmoid
	case "tanh":
		activation = network.ActivationTypeTanh
	default:
		flag.PrintDefaults()
		return runConfig{}, fmt.Errorf("unknown activation '%s'", *activationVal)
	}
	countStrs := strings.Split(*hiddenLayerCountsStr, ",")
	hiddenLayerCounts := make([]int, 0, len(countStrs))
	for _, s := range countStrs {
		v, err := strconv.ParseInt(strings.TrimSpace(s), 10, 32)
		if err != nil {
			return runConfig{}, fmt.Errorf("invalid layer count vaue '%s'", *hiddenLayerCountsStr)
		}
		hiddenLayerCounts = append(hiddenLayerCounts, int(v))
	}
	return runConfig{
		Action:      *action,
		ModelFile:   *model,
		DataSetFile: *dataset,
		Epochs:      *epochs,
		networkConfig: networkConfig{
			Activation:        activation,
			LearningRate:      *learningRate,
			RandomSeed:        *randomSeed,
			HiddenLayerCounts: hiddenLayerCounts,
		},
	}, nil
}
