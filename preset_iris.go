package main

import (
	"fmt"
	"strconv"
)

const (
	irisInputCount  = 4
	irisOutputCount = 3
)

func irisPreset(cfg *runConfig) error {
	cfg.InputCount = irisInputCount
	cfg.HiddenLayerCounts = []int{2}
	cfg.OutputCount = irisOutputCount
	cfg.TestLogBatch = 1
	cfg.TrainLogBatch = 100
	cfg.TestParseRecord = irisParseRecord
	cfg.TrainParseRecord = irisParseRecord
	cfg.Epochs = 200
	return nil
}

// irisColMax maps the input column to the maximum value in the training data so it can be normalized between 0 and 1
var irisColMax = map[int]float64{
	0: 8,
	1: 4.5,
	2: 7,
	3: 2.5,
}

// irisLabels maps the data label to the output neuron index
var irisLabels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

func irisParseRecord(record []string) (inputs, targets []float64, err error) {
	inputs = make([]float64, irisInputCount)
	for i := 0; i < irisInputCount; i++ { // ignore last column (which is the label)
		x, err := strconv.ParseFloat(record[i], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("parse input: %v", err)
		}
		max, ok := irisColMax[i]
		if !ok {
			return nil, nil, fmt.Errorf("invalid index %d", i)
		}
		inputs[i] = (x / max * 0.99) + 0.01
	}
	targets, err = irisTrainingTargets(record[irisInputCount]) // last column is the label
	if err != nil {
		return nil, nil, err
	}
	return inputs, targets, nil
}

func irisTrainingTargets(target string) ([]float64, error) {
	targets := make([]float64, irisOutputCount)
	for i := range targets {
		targets[i] = 0.01
	}
	x, ok := irisLabels[target]
	if !ok {
		return nil, fmt.Errorf("unknown target label '%s'", target)
	}
	targets[x] = 0.99
	return targets, nil
}
