package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"neural-net-go/network"
	"os"
	"strconv"
	"time"
)

func mnistRun() error {
	mnist := flag.String("mnist", "", "Action 'train' or 'test' against mnist dataset.")
	model := flag.String("model", "models/mnist.default.model", "File path of network model to load and save. If it doesn't exist a new network will be created.")
	dataset := flag.String("dataset", "", "File path of source dataset.")
	epochs := flag.Int("epochs", 1, "Number of training epochs. (Ignored if not training.)")
	flag.Parse()

	var n *network.Network
	if _, err := os.Stat(*model); err == nil {
		log.Printf("loading model from file %s...", *model)
		n, err = network.LoadFile(*model)
		if err != nil {
			return err
		}
		log.Printf("current network trained on %d records", n.Config().Trained)
	} else if os.IsNotExist(err) {
		log.Printf("no existing model file found at %s, creating new network with random weights...", *model)
		n = network.NewRandom(network.Config{
			LayerCounts: []int{784,100,10},
			Rate:   0.1,
			Seed:   0,
		})
	} else {
		return fmt.Errorf("checking model file: %v", err)
	}

	switch *mnist {
	case "train":
		if *dataset == "" {
			*dataset = "datasets/mnist_train.csv"
		}
		if err := mnistTrain(n, *epochs, *dataset); err != nil {
			return err
		}
		if err := n.SaveFile(*model); err != nil {
			return err
		}
	case "test":
		if *dataset == "" {
			*dataset = "datasets/mnist_test.csv"
		}
		if err := mnistTest(n, *dataset); err != nil {
			return err
		}
	default:
		return fmt.Errorf("invalid arguments: '-help' for more info")
	}

	return nil
}

func mnistTrain(net *network.Network, epochs int, filename string) error {
	start := time.Now()
	const mnistOutput = 10
	cfg := net.Config()
	if cfg.LayerCounts[2] != mnistOutput {
		return fmt.Errorf("mnist requires output of 10, got %d", cfg.LayerCounts[2])
	}
	log.Printf("training %d epochs", epochs)
	for e := 1; e <= epochs; e++ {
		if err := mnistTrainEpoch(net, e, filename, cfg); err != nil {
			return err
		}
	}
	log.Printf("time to train %d epochs: %v", epochs, time.Since(start))
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
	log.Printf("starting prediction test...")
	for {
		line++
		if line % logBatch == 0 {
			log.Printf("prediction test line %d...", line)
		}
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, cfg.LayerCounts[0])
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
		outputs := net.Predict(inputs)
		answer := 0
		highest := 0.0
		for i := 0; i < cfg.LayerCounts[2]; i++ {
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

	log.Printf("took %v to test", time.Since(start))
	log.Printf("scored %d/%d correct predictions: %0.2f%%", score, total, float32(score)*100/float32(total))

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
	const logBatch = 1000
	log.Printf("epoch %d: starting training...", e)
	for {
		line++
		if line % logBatch == 0 {
			log.Printf("epoch %d: training lines %d...", e, line)
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

		net.Train(inputs, targets)
	}
	return nil
}

func mnistTrainingInputs(record []string, cfg network.Config) ([]float64, error) {
	if len(record)-1 != cfg.LayerCounts[0] {
		return nil, fmt.Errorf("mismatched network inputs: need %d, got %d", len(record)-1, cfg.LayerCounts[0])
	}
	inputs := make([]float64, cfg.LayerCounts[0])
	for i := 1; i < cfg.LayerCounts[0]; i++ {
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