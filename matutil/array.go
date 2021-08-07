package matutil

import (
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

// OptRandomArraySource sets the source
func OptRandomArraySource(src rand.Source) func(*RandomArrayCfg) {
	return func(cfg *RandomArrayCfg) {
		cfg.Src = src
	}
}

// RandomArrayCfg settings for random array generation.
type RandomArrayCfg struct {
	Src rand.Source
}

// RandomArray returns a random array of size.
func RandomArray(size int, v float64, opts ...func(*RandomArrayCfg)) []float64 {
	sqrtV := math.Sqrt(v)
	cfg := RandomArrayCfg{}
	for _, opt := range opts {
		opt(&cfg)
	}
	dist := distuv.Uniform{
		Min: -1 / sqrtV,
		Max: 1 / sqrtV,
		Src: cfg.Src,
	}
	data := make([]float64, size)
	for i := size - 1; i >= 0; i-- {
		data[i] = dist.Rand()
	}
	return data
}
