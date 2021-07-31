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
	outputs, err := propagateForwards(inputs, n.weights)
	if err != nil {
		return nil, err
	}
	return outputs[len(outputs)-1], nil
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

	layerOutputs, err := propagateForwards(inputs, n.weights)
	if err != nil {
		return err
	}
	finalOutputs := layerOutputs[len(layerOutputs)-1]
	errors, err := findErrors(targets, finalOutputs, n.weights)
	if err != nil {
		return fmt.Errorf("finding errors: %v", err)
	}
	n.weights, err = propagateBackwards(n.weights, errors, layerOutputs, inputs, n.cfg.Rate)
	if err != nil {
		return err
	}

	n.cfg.Trained++

	return nil
}

// Trained returns the number of training runs.
func (n Network) Trained() uint64 {
	return n.cfg.Trained
}

func propagateBackwards(weights, errors, outputs []*mat.Dense, inputs mat.Matrix, rate float64) ([]*mat.Dense, error) {
	adjustedWeights := make([]*mat.Dense, len(weights))

	var err error
	for i := len(weights) - 1; i >= 1; i-- {
		adjustedWeights[i], err = backward(outputs[i], errors[i], weights[i], outputs[i-1], rate)
		if err != nil {
			return nil, err
		}
	}
	adjustedWeights[0], err = backward(outputs[0], errors[0], weights[0], inputs, rate)
	if err != nil {
		return nil, err
	}

	return adjustedWeights, nil
}

func propagateForwards(inputs mat.Matrix, weights []*mat.Dense) ([]*mat.Dense, error) {
	outputs := make([]*mat.Dense, 0, len(weights))
	for _, weight := range weights {
		layerOutput, err := forward(inputs, weight)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, layerOutput)
		inputs = layerOutput
	}
	return outputs, nil
}

func backward(outputs, errors, weights, inputs mat.Matrix, learningRate float64) (*mat.Dense, error) {
	actDer, err := sigmoidPrime(outputs)
	if err != nil {
		return nil, fmt.Errorf("applying activation derivative: %v", err)
	}
	multiply, err := matutil.Multiply(errors, actDer)
	if err != nil {
		return nil, fmt.Errorf("applying errors to activation derivative: %v", err)
	}
	dot, err := matutil.Dot(multiply, inputs.T())
	if err != nil {
		return nil, fmt.Errorf("applying activated errors to inputs: %v", err)
	}
	scale := matutil.Scale(learningRate, dot)
	adjusted := matutil.Add(weights, scale)
	return adjusted, nil
}

func findErrors(targets mat.Matrix, finalOutputs mat.Matrix, weights []*mat.Dense) ([]*mat.Dense, error) {
	errors := make([]*mat.Dense, len(weights))
	lastErrors, err := matutil.Subtract(targets, finalOutputs)
	if err != nil {
		return nil, fmt.Errorf("subtracting target from final outputs: %v", err)
	}
	for i := len(weights) - 1; i >= 1; i-- {
		errors[i] = lastErrors
		lastErrors, err = matutil.Dot(weights[i].T(), lastErrors)
		if err != nil {
			return nil, err
		}
	}
	errors[0] = lastErrors
	return errors, nil
}

func forward(inputs mat.Matrix, weights mat.Matrix) (*mat.Dense, error) {
	rawOutputs, err := matutil.Dot(weights, inputs)
	if err != nil {
		return nil, fmt.Errorf("applying weights: %v", err)
	}
	outputs, err := matutil.Apply(sigmoid, rawOutputs)
	if err != nil {
		return nil, fmt.Errorf("applying activation function: %v", err)
	}
	return outputs, nil
}

func sigmoid(_, _ int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}
func sigmoidPrime(m mat.Matrix) (*mat.Dense, error) {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	sub, err := matutil.Subtract(ones, m)
	if err != nil {
		return nil, err
	}
	return matutil.Multiply(m, sub) // m * (1 - m)
}
