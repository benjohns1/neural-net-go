package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"neural-net-go/matutil"
)

func main() {
	fmt.Println("ok")
	m := matutil.Scale(0.5, mat.NewDense(2, 2, []float64{0.5,1,2,3}))
	fmt.Printf("%+v", m)
}

type Network struct {
	inputs int
	hiddens int
	outputs int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate float64
}

type Config struct {
	input int
	hidden int
	output int
	rate float64
}
