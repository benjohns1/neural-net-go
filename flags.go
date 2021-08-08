package main

import (
	"flag"
	"fmt"
	"strconv"
	"strings"

	"github.com/benjohns1/neural-net-go/network"
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
	preset := flag.String("preset", "iris", "Preset 'mnist' or 'iris' dataset processing. Source dataset must be downloaded first, please see readme.")
	action := flag.String("action", "", "Action 'train' or 'test' against the dataset.")
	model := flag.String("model", "models/default.model", "File path of network model to load and save. If it doesn't exist a new network will be created.")
	dataset := flag.String("dataset", "", "File path of source dataset. (default \"datasets/{preset}_{action}.csv\")")
	epochs := flag.Int("epochs", 0, "Number of training epochs. Ignored if not training.")
	activationVal := flag.String("activation", "sigmoid", "Activation function 'sigmoid' or 'tanh'.")
	learningRate := flag.Float64("learning-rate", 0.1, "Network learning rate.")
	randomSeed := flag.Uint64("random-seed", 0, "Seed for random weight generation.")
	hiddenLayerCountsStr := flag.String("hidden-layer-counts", "", "Comma-separated list of neuron counts for hidden layers.")
	flag.Parse()
	if *dataset == "" {
		switch *action {
		case "train":
			fallthrough
		case "test":
			*dataset = fmt.Sprintf("datasets/%s_%s.csv", *preset, *action)
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
		trimmed := strings.TrimSpace(s)
		if trimmed == "" {
			continue
		}
		v, err := strconv.ParseInt(trimmed, 10, 32)
		if err != nil {
			return runConfig{}, fmt.Errorf("invalid layer count value '%s'", trimmed)
		}
		hiddenLayerCounts = append(hiddenLayerCounts, int(v))
	}
	cfg := runConfig{
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
	}

	cfgPreset := func(*runConfig) error { return nil }
	switch *preset {
	case "":
		break
	case "mnist":
		cfgPreset = mnistPreset
	case "iris":
		cfgPreset = irisPreset
	default:
		cfgPreset = func(*runConfig) error { return fmt.Errorf("unknown preset") }
	}
	if err := cfgPreset(&cfg); err != nil {
		return cfg, err
	}

	if cfg.Epochs == 0 {
		cfg.Epochs = 1
	}
	if len(cfg.HiddenLayerCounts) == 0 {
		cfg.HiddenLayerCounts = []int{cfg.InputCount}
	}

	return cfg, nil
}
