package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"neural-net-go/network"
	"neural-net-go/storage"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	cfg, err := parseCmdFlags()
	if err != nil {
		log.Fatalf("mnist config error: %v", err)
	}
	if err := csvRun(cfg); err != nil {
		log.Fatalf("mnist error: %v", err)
	}
}

type parseRecordFunc func(record []string) (inputs, targets []float64, err error)

func csvRun(cfg runConfig) error {
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
		log.Printf("No existing model file found at %s, creating new network with random weights seeded with %d...", cfg.ModelFile, cfg.RandomSeed)
		n, err = network.NewRandom(network.Config{
			InputCount:  cfg.InputCount,
			LayerCounts: append(cfg.HiddenLayerCounts, cfg.OutputCount),
			Rate:        cfg.LearningRate,
			RandSeed:    cfg.RandomSeed,
			Activation:  cfg.Activation,
		})
		if err != nil {
			return fmt.Errorf("creating new random network: %v", err)
		}
	} else {
		return fmt.Errorf("checking model file: %v", err)
	}

	switch cfg.Action {
	case "train":
		if err := train(n, cfg.Epochs, cfg.DataSetFile, cfg.TrainLogBatch, cfg.TrainParseRecord); err != nil {
			return err
		}
		if err := file.Save(n, cfg.ModelFile); err != nil {
			return err
		}
	case "test":
		if err := test(n, cfg.DataSetFile, cfg.TestLogBatch, cfg.TestParseRecord); err != nil {
			return err
		}
	default:
		return fmt.Errorf("invalid action: '-help' for more info")
	}

	return nil
}

func train(net *network.Network, epochs int, filename string, logBatch int, parseRecord parseRecordFunc) error {
	start := time.Now()
	cfg := net.Config()
	l := len(cfg.LayerCounts)
	if l == 0 {
		return fmt.Errorf("layer counts cannot be zero")
	}
	log.Printf("Training %d epochs", epochs)
	for e := 1; e <= epochs; e++ {
		if err := trainEpoch(net, e, filename, cfg, logBatch, parseRecord); err != nil {
			return err
		}
	}
	log.Printf("Time to train %d epochs: %v", epochs, time.Since(start))
	return nil
}

func test(net *network.Network, filename string, logBatch int, parseRecord parseRecordFunc) error {
	start := time.Now()
	checkFile, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("opening file: %v", err)
	}
	defer func() {
		_ = checkFile.Close()
	}()

	score := 0
	total := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	line := 0
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
		inputs, targets, err := parseRecord(record)
		if err != nil {
			return err
		}
		outputs, err := net.Predict(inputs)
		if err != nil {
			return fmt.Errorf("predicting: %v", err)
		}
		if getPrediction(outputs) == getTarget(targets) {
			score++
		}
		total++
	}

	log.Printf("Took %v to test", time.Since(start))
	log.Printf("Scored %d/%d correct predictions: %0.2f%%", score, total, float32(score)*100/float32(total))

	return nil
}

func getTarget(targets []float64) int {
	answer := 0
	best := 0.0
	for i, target := range targets {
		if target > best {
			answer = i
			best = target
		}
	}
	return answer
}

func getPrediction(outputs *mat.Dense) int {
	rows, _ := outputs.Dims()
	answer := 0
	best := 0.0
	for i := 0; i < rows; i++ {
		val := outputs.At(i, 0)
		if val > best {
			answer = i
			best = val
		}
	}
	return answer
}

func trainEpoch(net *network.Network, e int, filename string, cfg network.Config, logBatch int, parseRecord parseRecordFunc) error {
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
		inputs, targets, err := trainingInputs(parseRecord, cfg.InputCount, record)
		if err != nil {
			return fmt.Errorf("parsing training input: %v", err)
		}

		if err := net.Train(inputs, targets); err != nil {
			return fmt.Errorf("training: %v", err)
		}
	}
	return nil
}

func trainingInputs(parseRecord parseRecordFunc, count int, record []string) (inputs []float64, targets []float64, err error) {
	if len(record)-1 != count {
		return nil, nil, fmt.Errorf("mismatched inputs: %d record input values, expecting input count %d", len(record)-1, count)
	}
	return parseRecord(record)
}
