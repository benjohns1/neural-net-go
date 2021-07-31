package network

import (
	"fmt"
	"math"
	"neural-net-go/matutil"

	"gonum.org/v1/gonum/mat"
)

// Config network constructor.
type Config struct {
	InputCount  int
	LayerCounts []int
	Rate        float64
	RandSeed    uint64
	RandState   uint64
	Trained     uint64
}

// Network struct.
type Network struct {
	cfg     Config
	weights []*mat.Dense // hidden and output layers
}

// NewRandom constructs a new network with random weights from a config.
func NewRandom(cfg Config) (*Network, error) {
	src := Rand{cfg.RandSeed, cfg.RandState}.GetSource()
	weights := make([]*mat.Dense, 0, len(cfg.LayerCounts))
	count := cfg.InputCount
	for _, nextCount := range cfg.LayerCounts {
		weights = append(weights, mat.NewDense(nextCount, count, matutil.RandomArray(nextCount*count, float64(count), matutil.OptRandomArraySource(src))))
		count = nextCount
	}
	return New(cfg, weights)
}

// New constructs a new network with the specified layer weights.
func New(cfg Config, weights []*mat.Dense) (*Network, error) {
	if len(weights) != len(cfg.LayerCounts) {
		return nil, fmt.Errorf("layer weight count '%d' must be equal configured layer count '%d'", len(weights), len(cfg.LayerCounts))
	}
	previousCount := cfg.InputCount
	for i, weight := range weights {
		currentCount := cfg.LayerCounts[i]
		rc, cc := weight.Dims()
		if rc*cc != previousCount*currentCount {
			return nil, fmt.Errorf("layer %d size %d must equal layer 0 weight count %d", i, previousCount*currentCount, rc*cc)
		}
		previousCount = currentCount
	}
	return &Network{
		cfg:     cfg,
		weights: weights,
	}, nil
}

// Config gets the networks configuration.
func (n Network) Config() Config {
	return n.cfg
}

// Predict outputs from a trained network.
func (n Network) Predict(inputData []float64) (*mat.Dense, error) {
	inputs, err := matutil.FromVector(inputData)
	if err != nil {
		return nil, fmt.Errorf("creating matrix from input data: %v", err)
	}
	_, finalOutputs := propagateForwards(inputs, n.weights[0], n.weights[1])
	return finalOutputs, nil
}

// Train the network with a single set of inputs and target outputs.
func (n *Network) Train(input []float64, target []float64) error {
	inputs, err := matutil.FromVector(input)
	if err != nil {
		return fmt.Errorf("creating input matrix: %v", err)
	}
	targets, err := matutil.FromVector(target)
	if err != nil {
		return fmt.Errorf("creating target matrix: %v", err)
	}

	hiddenOutputs, finalOutputs := propagateForwards(inputs, n.weights[0], n.weights[1])
	outputErrors, hiddenErrors := findErrors(targets, finalOutputs, n.weights[1])
	n.weights[1], n.weights[0] = propagateBackwards(n.weights[1], finalOutputs, outputErrors, hiddenOutputs, n.weights[0], hiddenErrors, inputs, n.cfg.Rate)

	n.cfg.Trained++

	return nil
}

// Trained returns the number of training runs.
func (n Network) Trained() uint64 {
	return n.cfg.Trained
}

func propagateBackwards(outputWeights, finalOutputs, outputErrors, hiddenOutputs, hiddenWeights, hiddenErrors, inputs mat.Matrix, rate float64) (adjustedOutputWeights, adjustedHiddenWeights *mat.Dense) {
	adjustedOutputWeights = backward(finalOutputs, outputErrors, outputWeights, hiddenOutputs, rate)
	adjustedHiddenWeights = backward(hiddenOutputs, hiddenErrors, hiddenWeights, inputs, rate)
	return adjustedOutputWeights, adjustedHiddenWeights
}

func propagateForwards(inputs, hiddenWeights, outputWeights mat.Matrix) (hiddenOutputs, finalOutputs *mat.Dense) {
	hiddenOutputs = forward(inputs, hiddenWeights)
	finalOutputs = forward(hiddenOutputs, outputWeights)
	return hiddenOutputs, finalOutputs
}

func backward(outputs, errors, weights, inputs mat.Matrix, learningRate float64) *mat.Dense {
	multiply := matutil.Multiply(errors, sigmoidPrime(outputs))
	dot := matutil.Dot(multiply, inputs.T())
	scale := matutil.Scale(learningRate, dot)
	adjusted := matutil.Add(weights, scale)
	return adjusted
}

func findErrors(targets mat.Matrix, finalOutputs mat.Matrix, outputWeights mat.Matrix) (outputErrors, hiddenErrors *mat.Dense) {
	outputErrors = matutil.Subtract(targets, finalOutputs)
	hiddenErrors = matutil.Dot(outputWeights.T(), outputErrors)
	return outputErrors, hiddenErrors
}

func forward(inputs mat.Matrix, inputWeights mat.Matrix) *mat.Dense {
	rawOutputs := matutil.Dot(inputWeights, inputs)
	return matutil.Apply(sigmoid, rawOutputs)
}

func sigmoid(_, _ int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}
func sigmoidPrime(m mat.Matrix) *mat.Dense {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return matutil.Multiply(m, matutil.Subtract(ones, m)) // m * (1 - m)
}
