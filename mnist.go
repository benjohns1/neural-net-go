package main

import (
	"fmt"
	"strconv"
)

const (
	mnistInputCount  = 784
	mnistOutputCount = 10
	testLogBatch     = 1000
	trainLogBatch    = 10000
)

func mnistRun(cfg runConfig) error {
	cfg.InputCount = mnistInputCount
	cfg.OutputCount = mnistOutputCount
	cfg.TestLogBatch = testLogBatch
	cfg.TrainLogBatch = trainLogBatch
	cfg.TestParseRecord = parseMnistRecord
	cfg.TrainParseRecord = parseMnistRecord
	return run(cfg)
}

type parseRecordFunc func(count int, record []string) (inputs, targets []float64, err error)

func parseMnistRecord(count int, record []string) (inputs, targets []float64, err error) {
	inputs = make([]float64, count)
	for i := range inputs {
		x, err := strconv.ParseFloat(record[i+1], 64) // ignore first column (which is the label)
		if err != nil {
			return nil, nil, fmt.Errorf("parse input: %v", err)
		}
		inputs[i] = (x / 255.0 * 0.99) + 0.01
	}
	targets, err = mnistTrainingTargets(record[0])
	if err != nil {
		return nil, nil, err
	}
	return inputs, targets, nil
}

func mnistTrainingTargets(target string) ([]float64, error) {
	targets := make([]float64, mnistOutputCount)
	for i := range targets {
		targets[i] = 0.01
	}
	x, err := strconv.Atoi(target)
	if err != nil {
		return nil, fmt.Errorf("target parse: %v", err)
	}
	targets[x] = 0.99
	return targets, nil
}
