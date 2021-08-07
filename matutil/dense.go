package matutil

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Dot product of 2 matrices.
func Dot(m, n mat.Matrix) (d *mat.Dense, err error) {
	var o mat.Dense
	if err := safe(func() error {
		o.Product(m, n)
		return nil
	}); err != nil {
		return nil, err
	}
	return &o, nil
}

// Apply a function to all elements of a matrix.
func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) (d *mat.Dense, err error) {
	var o mat.Dense
	if err := safe(func() error {
		o.Apply(fn, m)
		return nil
	}); err != nil {
		return nil, err
	}
	return &o, nil
}

// Scale each element of a matrix.
func Scale(s float64, m mat.Matrix) (*mat.Dense, error) {
	var o mat.Dense
	if err := safe(func() error {
		o.Scale(s, m)
		return nil
	}); err != nil {
		return nil, err
	}
	return &o, nil
}

// MulElem each corresponding element of the matrices together.
func MulElem(m, n mat.Matrix) (*mat.Dense, error) {
	var o mat.Dense
	if err := safe(func() error {
		o.MulElem(m, n)
		return nil
	}); err != nil {
		return nil, err
	}
	return &o, nil
}

// Add each corresponding element of the matrices together.
func Add(m, n mat.Matrix) (*mat.Dense, error) {
	var o mat.Dense
	if err := safe(func() error {
		o.Add(m, n)
		return nil
	}); err != nil {
		return nil, err
	}
	return &o, nil
}

// Sub the corresponding elements of the second matrix from the first.
func Sub(m, n mat.Matrix) (*mat.Dense, error) {
	var o mat.Dense
	if err := safe(func() error {
		o.Sub(m, n)
		return nil
	}); err != nil {
		return nil, err
	}
	return &o, nil
}

// FromVector creates a single-column matrix from a vector.
func FromVector(v []float64) (*mat.Dense, error) {
	l := len(v)
	if l == 0 {
		return nil, fmt.Errorf("vector length is zero, cannot create matrix")
	}
	return mat.NewDense(l, 1, v), nil
}

// ToVector creates a vector from a single-column matrix.
func ToVector(m mat.Matrix) (v []float64, err error) {
	if m == nil {
		return nil, fmt.Errorf("matrix cannot be nil")
	}
	if err := safe(func() error {
		r, c := m.Dims()
		if c != 1 {
			return fmt.Errorf("matrix must have a single column to convert to a vector, but has %d", c)
		}
		v = make([]float64, r)
		for i := r - 1; i >= 0; i-- {
			v[i] = m.At(i, 0)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	return v, nil
}

func safe(f func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%s", r)
		}
	}()
	return f()
}
