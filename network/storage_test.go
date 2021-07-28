package network_test

import (
	"neural-net-go/network"
	"reflect"
	"testing"
)

func TestLoadFile(t *testing.T) {
	tests := []struct {
		name    string
		save network.Network
		assert func(*testing.T, *network.Network)
		wantErr bool
	}{
		{
			name: "should load from saved file",
			save: network.NewRandom(network.Config{
				Input:  3,
				Hidden: 2,
				Output: 1,
				Seed:   0,
			}),
			assert: func(t *testing.T, n *network.Network) {
				p := n.Predict([]float64{0,0,0})
				got := p.RawMatrix().Data
				want := []float64{0.5864282096612484}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("Load() assert got = %+v, want %+v", got, want)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.save.SaveFile("network.model"); err != nil {
				t.Fatal(err)
			}
			got, err := network.LoadFile("network.model")
			if (err != nil) != tt.wantErr {
				t.Errorf("Load() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.assert != nil {
				tt.assert(t, got)
			}
		})
	}
}
