package matutil

import "gonum.org/v1/gonum/mat"

// Dot product of 2 matrices.
func Dot(m, n mat.Matrix) *mat.Dense {
	var o mat.Dense
	o.Product(m, n)
	return &o
}

// Apply a function to all elements of a matrix.
func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) *mat.Dense {
	var o mat.Dense
	o.Apply(fn, m)
	return &o
}

// Scale each element of a matrix.
func Scale(s float64, m mat.Matrix) *mat.Dense {
	var o mat.Dense
	o.Scale(s, m)
	return &o
}

// Multiply each corresponding element of the matrices together.
func Multiply(m, n mat.Matrix) *mat.Dense {
	var o mat.Dense
	o.MulElem(m, n)
	return &o
}

// Add each corresponding element of the matrices together.
func Add(m, n mat.Matrix) *mat.Dense {
	var o mat.Dense
	o.Add(m, n)
	return &o
}

// Subtract the corresponding elements of the second matrix from the first.
func Subtract(m, n mat.Matrix) *mat.Dense {
	var o mat.Dense
	o.Sub(m, n)
	return &o
}

// AddScalar adds the value to each element in the matrix.
func AddScalar(i float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return Add(m, n)
}