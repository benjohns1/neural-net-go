package main

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"neural-net-go/matutil"
	"time"
)

func main() {
	seed := uint64(time.Now().UnixNano())
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

type Network struct {
	cfg Config
	src rand.Source
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
}

type Config struct {
	input int
	hidden int
	output int
	rate float64
	seed uint64
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
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := matutil.Dot(n.hiddenWeights, inputs)
	hiddenOutputs := matutil.Apply(sigmoid, hiddenInputs)
	finalInputs := matutil.Dot(n.outputWeights, hiddenOutputs)
	return matutil.Apply(sigmoid, finalInputs)
}

func (n *Network) Train(inputData []float64, targetData []float64) {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := matutil.Dot(n.hiddenWeights, inputs)
	hiddenOutputs := matutil.Apply(sigmoid, hiddenInputs)
	finalInputs := matutil.Dot(n.outputWeights, hiddenOutputs)
	finalOutputs := matutil.Apply(sigmoid, finalInputs)

	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := matutil.Subtract(targets, finalOutputs)
	hiddenErrors := matutil.Dot(n.outputWeights.T(), outputErrors)

	// backpropagate
	n.outputWeights = matutil.Add(n.outputWeights,
		matutil.Scale(n.cfg.rate,
			matutil.Dot(matutil.Multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T())))

	n.hiddenWeights = matutil.Add(n.hiddenWeights,
		matutil.Scale(n.cfg.rate,
			matutil.Dot(matutil.Multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T())))
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