package activation

import (
	"math"
	"neural-net-go/matutil"

	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct{}

// Value computes the sigmoid for activation of a single value.
func (s Sigmoid) Value(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

// MatrixDerivative assumes the given output values are sigmoid(v), then this function
//  computes sigmoidPrime as sigmoid(v) * (1 - sigmoid(v))
func (s Sigmoid) MatrixDerivative(outputs mat.Matrix) (*mat.Dense, error) {
	rows, _ := outputs.Dims()
	ones := mat.NewDense(rows, 1, matutil.FillArray(rows, 1))
	sub, err := matutil.Sub(ones, outputs)
	if err != nil {
		return nil, err
	}
	return matutil.MulElem(outputs, sub)
}
