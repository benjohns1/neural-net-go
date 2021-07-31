package network_test

import (
	"neural-net-go/matutil"
	"neural-net-go/network"
	"reflect"
	"testing"
)

func TestNetwork_Predict(t *testing.T) {
	type args struct {
		inputData []float64
	}
	tests := []struct {
		name    string
		n       *network.Network
		args    args
		want    []float64
		wantErr bool
	}{
		{
			name: "should error due to nil input data",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				inputData: nil,
			},
			wantErr: true,
		},
		{
			name: "should error due to empty input data",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				inputData: []float64{},
			},
			wantErr: true,
		},
		{
			name: "should error due to input data length not matching input count",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				inputData: []float64{1},
			},
			wantErr: true,
		},
		{
			name: "should successfully output an expected value from a network generated from a seed",
			n: func() *network.Network {
				n, err := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				if err != nil {
					t.Fatal(err)
				}
				return n
			}(),
			args: args{
				inputData: []float64{1, 2, 3},
			},
			want: []float64{0.6348888771252147},
		},
		{
			name: "should successfully output an expected value from a 4-layer network generated from a seed",
			n: func() *network.Network {
				n, err := network.NewRandom(network.Config{
					InputCount:  4,
					LayerCounts: []int{3, 2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				if err != nil {
					t.Fatal(err)
				}
				return n
			}(),
			args: args{
				inputData: []float64{1, 2, 3, 4},
			},
			want: []float64{0.49688797374514093},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.n.Predict(tt.args.inputData)
			if (err != nil) != tt.wantErr {
				t.Errorf("Predict() error = %v, wantErr %v", err, tt.wantErr)
			}
			var gotVector []float64
			if got != nil {
				gotVector, err = matutil.ToVector(got)
				if err != nil {
					t.Fatal(err)
				}
			}
			if !reflect.DeepEqual(gotVector, tt.want) {
				t.Errorf("Predict() gotVector = %v, want %v", gotVector, tt.want)
			}
		})
	}
}

func TestNetwork_Train(t *testing.T) {
	type args struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name             string
		n                *network.Network
		args             args
		wantErr          bool
		wantTrainedCount uint64
	}{
		{
			name: "should error due to nil input data",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input: nil,
			},
			wantErr: true,
		},
		{
			name: "should error due to empty input data",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input: []float64{},
			},
			wantErr: true,
		},
		{
			name: "should error due to input data length not matching input count",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input: []float64{1},
			},
			wantErr: true,
		},
		{
			name: "should error due to nil target data",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input:  []float64{1, 2, 3},
				target: nil,
			},
			wantErr: true,
		},
		{
			name: "should error due to empty target data",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input:  []float64{1, 2, 3},
				target: []float64{},
			},
			wantErr: true,
		},
		{
			name: "should error due to target data length not matching output count",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input:  []float64{1, 2, 3},
				target: []float64{1, 2, 3},
			},
			wantErr: true,
		},
		{
			name: "should successfully train 1 record",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input:  []float64{1, 2, 3},
				target: []float64{1},
			},
			wantTrainedCount: 1,
		},
		{
			name: "should successfully train a 4-layer network with 1 record",
			n: func() *network.Network {
				n, _ := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 2, 1},
					Rate:        0.1,
					RandSeed:    0,
				})
				return n
			}(),
			args: args{
				input:  []float64{1, 2, 3},
				target: []float64{1},
			},
			wantTrainedCount: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.n.Train(tt.args.input, tt.args.target); (err != nil) != tt.wantErr {
				t.Errorf("Train() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.n.Trained() != tt.wantTrainedCount {
				t.Errorf("Train() trained count = %v, wantTrainedCount %v", tt.n.Trained(), tt.wantTrainedCount)
			}
		})
	}
}

func TestNetwork_PredictFromTrained(t *testing.T) {
	type args struct {
		inputData []float64
	}
	type trainingRecord struct {
		data   []float64
		target []float64
	}
	tests := []struct {
		name         string
		n            *network.Network
		trainingData []trainingRecord
		args         args
		want         []float64
		wantErr      bool
	}{
		{
			name: "should successfully output an expected value from a network trained by a single record",
			n: func() *network.Network {
				n, err := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 1},
					Rate:        1,
					RandSeed:    0,
				})
				if err != nil {
					t.Fatal(err)
				}
				return n
			}(),
			trainingData: []trainingRecord{
				{[]float64{1, 2, 3}, []float64{1}},
			},
			args: args{
				inputData: []float64{1, 2, 3},
			},
			want: []float64{0.6724294088734996},
		},
		{
			name: "should successfully output an expected value from a 4-layer network trained by a single record",
			n: func() *network.Network {
				n, err := network.NewRandom(network.Config{
					InputCount:  3,
					LayerCounts: []int{2, 2, 1},
					Rate:        1,
					RandSeed:    0,
				})
				if err != nil {
					t.Fatal(err)
				}
				return n
			}(),
			trainingData: []trainingRecord{
				{[]float64{1, 2, 3}, []float64{1}},
			},
			args: args{
				inputData: []float64{1, 2, 3},
			},
			want: []float64{0.45405359950545143},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, datum := range tt.trainingData {
				if err := tt.n.Train(datum.data, datum.target); err != nil {
					t.Fatal(err)
				}
			}
			got, err := tt.n.Predict(tt.args.inputData)
			if (err != nil) != tt.wantErr {
				t.Errorf("Predict() error = %v, wantErr %v", err, tt.wantErr)
			}
			var gotVector []float64
			if got != nil {
				gotVector, err = matutil.ToVector(got)
				if err != nil {
					t.Fatal(err)
				}
			}
			if !reflect.DeepEqual(gotVector, tt.want) {
				t.Errorf("Predict() gotVector = %v, want %v", gotVector, tt.want)
			}
		})
	}
}
