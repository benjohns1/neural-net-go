package network

import (
	"fmt"
	"math"
	"neural-net-go/matutil"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// Config network constructor.
type Config struct {
	LayerCounts []int
	Rate        float64
	Seed        uint64
	Trained     uint64
}

// Network struct.
type Network struct {
	cfg           Config
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
}

// NewRandom constructs a new network with random weights from a config.
func NewRandom(cfg Config) *Network {
	src := rand.NewSource(cfg.Seed)
	weights := []*mat.Dense{
		mat.NewDense(cfg.LayerCounts[1], cfg.LayerCounts[0], matutil.RandomArray(cfg.LayerCounts[0]*cfg.LayerCounts[1], float64(cfg.LayerCounts[0]), matutil.OptRandomArraySource(src))),
		mat.NewDense(cfg.LayerCounts[2], cfg.LayerCounts[1], matutil.RandomArray(cfg.LayerCounts[1]*cfg.LayerCounts[2], float64(cfg.LayerCounts[1]), matutil.OptRandomArraySource(src))),
	}
	return New(cfg, weights)
}

// New constructs a new network with the specified layer weights.
func New(cfg Config, weights []*mat.Dense) *Network {
	weightLen := len(weights)
	if weightLen != 2 {
		panic("network weights must be 2 (until multiple layers are implemented)")
	}
	ri, ci := weights[0].Dims()
	if ri*ci != cfg.LayerCounts[0]*cfg.LayerCounts[1] {
		panic(fmt.Sprintf("hidden size %d doesn't match layer 0 weight count %d", cfg.LayerCounts[0]*cfg.LayerCounts[1], ri*ci))
	}
	rh, ch := weights[1].Dims()
	if rh*ch != cfg.LayerCounts[1]*cfg.LayerCounts[2] {
		panic(fmt.Sprintf("output size %d doesn't match layer 1 weight count %d", cfg.LayerCounts[1]*cfg.LayerCounts[2], rh*ch))
	}
	return &Network{
		cfg:           cfg,
		hiddenWeights: weights[0],
		outputWeights: weights[1],
	}
}

// Config gets the networks configuration.
func (n Network) Config() Config {
	return n.cfg
}

// Predict outputs from a trained network.
func (n Network) Predict(inputData []float64) *mat.Dense {
	inputs := matutil.FromVector(inputData)
	_, finalOutputs := propagateForwards(inputs, n.hiddenWeights, n.outputWeights)
	return finalOutputs
}

// Train the network with a single set of inputs and target outputs.
func (n *Network) Train(input []float64, target []float64) {
	inputs := matutil.FromVector(input)
	targets := matutil.FromVector(target)

	hiddenOutputs, finalOutputs := propagateForwards(inputs, n.hiddenWeights, n.outputWeights)
	outputErrors, hiddenErrors := findErrors(targets, finalOutputs, n.outputWeights)
	n.outputWeights, n.hiddenWeights = propagateBackwards(n.outputWeights, finalOutputs, outputErrors, hiddenOutputs, n.hiddenWeights, hiddenErrors, inputs, n.cfg.Rate)

	n.cfg.Trained++
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
