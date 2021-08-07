package activation

import (
	"math"
	"neural-net-go/matutil"

	"gonum.org/v1/gonum/mat"
)

type Tanh struct{}

// Value computes the hyperbolic tangent for activation of a single value.
func (s Tanh) Value(v float64) float64 {
	return math.Tanh(v)
}

func square(_, _ int, v float64) float64 {
	return math.Pow(v, 2)
}

// MatrixDerivative assumes the given output values are tanh(v), then this function
//  computes tanhPrime as 1 - tanh(v)^2
func (s Tanh) MatrixDerivative(outputs mat.Matrix) (*mat.Dense, error) {
	squared, err := matutil.Apply(square, outputs)
	if err != nil {
		return nil, err
	}
	rows, _ := outputs.Dims()
	ones := mat.NewDense(rows, 1, matutil.FillArray(rows, 1))
	return matutil.Sub(ones, squared)
}
