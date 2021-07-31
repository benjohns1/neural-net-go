package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"neural-net-go/network"
	"neural-net-go/storage"
	"os"
	"strconv"
	"time"
)

type mnistRunConfig struct {
	Action      string
	ModelFile   string
	DataSetFile string
	Epochs      int
}

func mnistParseCmdFlags() mnistRunConfig {
	action := flag.String("action", "", "Action 'train' or 'test' against the dataset.")
	model := flag.String("model", "models/mnist.default.model", "File path of network model to load and save. If it doesn't exist a new network will be created.")
	dataset := flag.String("dataset", "", "File path of source dataset. (Defaults to datasets/mnist_*.csv where * is the action chosen.)")
	epochs := flag.Int("epochs", 1, "Number of training epochs; ignored if not training.")
	flag.Parse()
	if *dataset == "" {
		switch *action {
		case "train":
			*dataset = "datasets/mnist_train.csv"
		case "test":
			*dataset = "datasets/mnist_test.csv"
		default:
			panic(fmt.Sprintf("unknown action '%s'", *action))
		}
	}
	return mnistRunConfig{
		Action:      *action,
		ModelFile:   *model,
		DataSetFile: *dataset,
		Epochs:      *epochs,
	}
}

func mnistRun(cfg mnistRunConfig) error {
	file := storage.NewJSONFile()

	var n *network.Network
	if _, err := os.Stat(cfg.ModelFile); err == nil {
		log.Printf("Loading model from file %s...", cfg.ModelFile)
		n = &network.Network{}
		err = file.Load(n, cfg.ModelFile)
		if err != nil {
			return err
		}
		log.Printf("Current network trained on %d records", n.Config().Trained)
	} else if os.IsNotExist(err) {
		var seed uint64
		log.Printf("No existing model file found at %s, creating new network with random weights seeded with %d...", cfg.ModelFile, seed)
		n, err = network.NewRandom(network.Config{
			InputCount:  784,
			LayerCounts: []int{100, 10},
			Rate:        0.1,
			RandSeed:    seed,
		})
		if err != nil {
			return fmt.Errorf("creating new random network: %v", err)
		}
	} else {
		return fmt.Errorf("checking model file: %v", err)
	}

	switch cfg.Action {
	case "train":
		if err := mnistTrain(n, cfg.Epochs, cfg.DataSetFile); err != nil {
			return err
		}
		if err := file.Save(n, cfg.ModelFile); err != nil {
			return err
		}
	case "test":
		if err := mnistTest(n, cfg.DataSetFile); err != nil {
			return err
		}
	default:
		return fmt.Errorf("invalid action: '-help' for more info")
	}

	return nil
}

func mnistTrain(net *network.Network, epochs int, filename string) error {
	start := time.Now()
	const mnistOutput = 10
	cfg := net.Config()
	l := len(cfg.LayerCounts)
	if l == 0 {
		return fmt.Errorf("layer counts cannot be zero")
	}
	if cfg.LayerCounts[len(cfg.LayerCounts)-1] != mnistOutput {
		return fmt.Errorf("mnist requires output of 10, got %d", cfg.LayerCounts[2])
	}
	log.Printf("Training %d epochs", epochs)
	for e := 1; e <= epochs; e++ {
		if err := mnistTrainEpoch(net, e, filename, cfg); err != nil {
			return err
		}
	}
	log.Printf("Time to train %d epochs: %v", epochs, time.Since(start))
	return nil
}

func mnistTest(net *network.Network, filename string) error {
	start := time.Now()
	checkFile, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("opening file: %v", err)
	}
	defer func() {
		_ = checkFile.Close()
	}()

	cfg := net.Config()
	score := 0
	total := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	line := 0
	const logBatch = 1000
	log.Printf("Starting prediction test...")
	for {
		line++
		if line%logBatch == 0 {
			log.Printf("Prediction test line %d...", line)
		}
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, cfg.InputCount)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return fmt.Errorf("parse input: %v", err)
			}
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}
		outputs, err := net.Predict(inputs)
		if err != nil {
			return fmt.Errorf("predicting: %v", err)
		}
		rows, _ := outputs.Dims()
		answer := 0
		highest := 0.0
		for i := 0; i < rows; i++ {
			val := outputs.At(i, 0)
			if val > highest {
				answer = i
				highest = val
			}
		}
		target, err := strconv.Atoi(record[0])
		if err != nil {
			return fmt.Errorf("parse record: %v", err)
		}
		if answer == target {
			score++
		}
		total++
	}

	log.Printf("Took %v to test", time.Since(start))
	log.Printf("Scored %d/%d correct predictions: %0.2f%%", score, total, float32(score)*100/float32(total))

	return nil
}

func mnistTrainEpoch(net *network.Network, e int, filename string, cfg network.Config) error {
	testFile, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("opening file: %v", err)
	}
	defer func() {
		_ = testFile.Close()
	}()
	r := csv.NewReader(bufio.NewReader(testFile))
	line := 0
	batchStart := time.Now()
	const logBatch = 10000
	log.Printf("Epoch %d: training first %d records...", e, logBatch)
	for {
		line++
		if line%logBatch == 0 {
			log.Printf("Epoch %d: last batch took %v, training next %d records from line %d...", e, time.Since(batchStart), logBatch, line)
			batchStart = time.Now()
		}
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs, err := mnistTrainingInputs(record, cfg)
		if err != nil {
			return err
		}

		targets, err := mnistTrainingTargets(record[0])
		if err != nil {
			return err
		}

		if err := net.Train(inputs, targets); err != nil {
			return fmt.Errorf("training: %v", err)
		}
	}
	return nil
}

func mnistTrainingInputs(record []string, cfg network.Config) ([]float64, error) {
	if len(record)-1 != cfg.InputCount {
		return nil, fmt.Errorf("mismatched network inputs: need %d, got %d", len(record)-1, cfg.InputCount)
	}
	inputs := make([]float64, cfg.InputCount)
	for i := 1; i < cfg.InputCount; i++ {
		x, err := strconv.ParseFloat(record[i], 64)
		if err != nil {
			return nil, fmt.Errorf("record parse: %v", err)
		}
		inputs[i] = (x / 255.0 * 0.99) + 0.01
	}
	return inputs, nil
}

func mnistTrainingTargets(target string) ([]float64, error) {
	targets := make([]float64, 10)
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
