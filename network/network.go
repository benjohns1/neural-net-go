package network

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"io/ioutil"
	"math"
	"neural-net-go/matutil"
	"os"
	"strings"
)

// Config network constructor.
type Config struct {
	Input  int
	Hidden int
	Output int
	Rate   float64
	Seed   uint64
}

// Network struct.
type Network struct {
	cfg           Config
	src           rand.Source
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
}

// New constructs a new network from a config.
func New(cfg Config) Network {
	src := rand.NewSource(cfg.Seed)
	return Network{
		cfg:           cfg,
		src:           src,
		hiddenWeights: mat.NewDense(cfg.Hidden, cfg.Input, randomArray(cfg.Input*cfg.Hidden, float64(cfg.Input), src)),
		outputWeights: mat.NewDense(cfg.Output, cfg.Hidden, randomArray(cfg.Hidden*cfg.Output, float64(cfg.Hidden), src)),
	}
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

func randomArray(size int, v float64, src rand.Source) []float64 {
	sqrtV := math.Sqrt(v)
	dist := distuv.Uniform{
		Min: -1 / sqrtV,
		Max: 1 / sqrtV,
		Src: src,
	}
	data := make([]float64, size)
	for i := size - 1; i >= 0; i-- {
		data[i] = dist.Rand()
	}
	return data
}

type storage struct {
	Version uint32 `json:"v"`
	Config `json:"c"`
	Layers []jsonMatrix `json:"l"`
}

type jsonMatrix struct {
	M *mat.Dense
}

func (m *jsonMatrix) MarshalJSON() ([]byte, error) {
	d, err := m.M.MarshalBinary()
	if err != nil {
		return d, fmt.Errorf("marshaling matrix: %v", err)
	}
	b64 := base64.StdEncoding.EncodeToString(d)
	return []byte(fmt.Sprintf("\"%s\"", b64)), nil
}

func (m *jsonMatrix) UnmarshalJSON(data []byte) error {
	dec, err := base64.StdEncoding.DecodeString(string(data))
	if err != nil {
		return fmt.Errorf("base64 decoding: %v", err)
	}
	trimmed := strings.Trim(string(dec), "\"")
	if err := m.M.UnmarshalBinary([]byte(trimmed)); err != nil {
		return fmt.Errorf("unmarshaling matrix: %v", err)
	}
	return nil
}

// Save a network to disk.
func (n Network) Save() error {
	s := storage{
		Version: 1,
		Config: n.cfg,
		Layers: []jsonMatrix{
			{M: n.hiddenWeights},
			{M: n.outputWeights},
		},
	}
	data, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("json marshaling: %v", err)
	}

	if err := ioutil.WriteFile("network.model", data, os.ModePerm); err != nil {
		return fmt.Errorf("writing file: %v", err)
	}
	return nil
}


//// Load a network from disk.
//func Load() *Network {
//
//}