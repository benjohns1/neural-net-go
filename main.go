package main

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"neural-net-go/matutil"
)

func main() {
	var seed uint64
	//seed = uint64(time.Now().UnixNano())
	nn := NewNetwork(Config{
		input:  3,
		hidden: 4,
		output: 2,
		rate:   0.01,
		seed:   seed,
	})
	nn.Train([]float64{1,2,3}, []float64{0,1})
	p := nn.Predict([]float64{1,2,3})
	fmt.Printf("seed: %d, %+v", seed, p)
}

type Config struct {
	input int
	hidden int
	output int
	rate float64
	seed uint64
}

type Network struct {
	cfg Config
	src rand.Source
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
}

func NewNetwork(cfg Config) Network {
	src := rand.NewSource(cfg.seed)
	return Network{
		cfg:           cfg,
		src:           src,
		hiddenWeights: mat.NewDense(cfg.hidden, cfg.input, randomArray(cfg.input*cfg.hidden, float64(cfg.input), src)),
		outputWeights: mat.NewDense(cfg.output, cfg.hidden, randomArray(cfg.hidden*cfg.output, float64(cfg.hidden), src)),
	}
}

func (n Network) Predict(inputData []float64) *mat.Dense {
	inputs := vectorToMatrix(inputData)
	_, finalOutputs := propagateForwards(inputs, n.hiddenWeights, n.outputWeights)
	return finalOutputs
}

func (n *Network) Train(inputData []float64, targetData []float64) {
	inputs := vectorToMatrix(inputData)
	targets := vectorToMatrix(targetData)

	hiddenOutputs, finalOutputs := propagateForwards(inputs, n.hiddenWeights, n.outputWeights)
	outputErrors, hiddenErrors := findErrors(targets, finalOutputs, n.outputWeights)
	n.outputWeights, n.hiddenWeights = propagateBackwards(n.outputWeights, finalOutputs, outputErrors, hiddenOutputs, n.hiddenWeights, hiddenErrors, inputs, n.cfg.rate)
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
	for i := size-1; i >= 0; i-- {
		data[i] = dist.Rand()
	}
	return data
}

func vectorToMatrix(v []float64) *mat.Dense {
	return mat.NewDense(len(v), 1, v)
}