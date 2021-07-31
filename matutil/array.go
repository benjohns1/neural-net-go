package matutil

import (
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

func OptRandomArraySource(src rand.Source) func(*distuv.Uniform) {
	return func(d *distuv.Uniform) {
		d.Src = src
	}
}

// RandomArray returns a random array of size.
func RandomArray(size int, v float64, opts ...func(*distuv.Uniform)) []float64 {
	sqrtV := math.Sqrt(v)
	dist := distuv.Uniform{
		Min: -1 / sqrtV,
		Max: 1 / sqrtV,
	}
	for _, opt := range opts {
		opt(&dist)
	}
	data := make([]float64, size)
	for i := size - 1; i >= 0; i-- {
		data[i] = dist.Rand()
	}
	return data
}
