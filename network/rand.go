package network

import "golang.org/x/exp/rand"

// Rand allows pseudo-random number generation that can be restored to a specific state for reproducible generation.
type Rand struct {
	Seed  uint64
	State uint64
}

// GetSource returns a new seeded source at the current state.
func (r Rand) GetSource() rand.Source {
	src := rand.NewSource(r.Seed)
	for i := r.State; i > 0; i-- {
		src.Uint64()
	}
	return src
}
