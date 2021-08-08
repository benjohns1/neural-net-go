package network_test

import (
	"testing"

	"github.com/benjohns1/neural-net-go/network"
)

func TestRand_GetSource(t *testing.T) {
	tests := []struct {
		name      string
		r1        network.Rand
		r2        network.Rand
		wantEqual bool
	}{
		{
			name:      "should produce the same random number from zero values",
			r1:        network.Rand{},
			r2:        network.Rand{},
			wantEqual: true,
		},
		{
			name:      "should produce the same random number from non-zero seeds",
			r1:        network.Rand{Seed: 100},
			r2:        network.Rand{Seed: 100},
			wantEqual: true,
		},
		{
			name:      "should produce different random numbers from different seeds",
			r1:        network.Rand{Seed: 1},
			r2:        network.Rand{Seed: 2},
			wantEqual: false,
		},
		{
			name:      "should produce the same random number from non-zero state",
			r1:        network.Rand{State: 100},
			r2:        network.Rand{State: 100},
			wantEqual: true,
		},
		{
			name:      "should produce the same random number from non-zero seeds and state",
			r1:        network.Rand{Seed: 100, State: 100},
			r2:        network.Rand{Seed: 100, State: 100},
			wantEqual: true,
		},
		{
			name:      "should produce different random numbers from the same seed with a different state",
			r1:        network.Rand{Seed: 100, State: 888},
			r2:        network.Rand{Seed: 100, State: 999},
			wantEqual: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got1 := tt.r1.GetSource().Uint64()
			got2 := tt.r2.GetSource().Uint64()
			if (got1 == got2) != tt.wantEqual {
				t.Errorf("GetSource() got1 = %v got2 = %v, wantEqual %v", got1, got2, tt.wantEqual)
			}
		})
	}
}
