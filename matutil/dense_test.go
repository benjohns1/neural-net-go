package matutil

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDot(t *testing.T) {
	type args struct {
		m mat.Matrix
		n mat.Matrix
	}
	tests := []struct {
		name      string
		args      args
		want      *mat.Dense
		wantPanic bool
	}{
		{
			name: "should calculate dot product for 1x1 matrices",
			args: args{
				m: mat.NewDense(1, 1, []float64{2}),
				n: mat.NewDense(1, 1, []float64{3}),
			},
			want: mat.NewDense(1, 1, []float64{6}),
		},
		{
			name: "should calculate dot product for 2x1 and 1x1 matrices",
			args: args{
				m: mat.NewDense(2, 1, []float64{1, 2}),
				n: mat.NewDense(1, 1, []float64{3}),
			},
			want: mat.NewDense(2, 1, []float64{3, 6}),
		},
		{
			name: "should panic if matrix dimensions to not allow dot product",
			args: args{
				m: mat.NewDense(2, 1, []float64{1, 2}),
				n: mat.NewDense(2, 1, []float64{3, 4}),
			},
			wantPanic: true,
		},
		{
			name: "should calculate dot product for 2x1 and 1x2 matrices",
			args: args{
				m: mat.NewDense(2, 1, []float64{1, 2}),
				n: mat.NewDense(1, 2, []float64{3, 4}),
			},
			want: mat.NewDense(2, 2, []float64{3, 4, 6, 8}),
		},
		{
			name: "should calculate dot product for 2x2 matrices",
			args: args{
				m: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				n: mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
			},
			want: mat.NewDense(2, 2, []float64{1*5 + 2*7, 1*6 + 2*8, 3*5 + 4*7, 3*6 + 4*8}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); (r != nil) != tt.wantPanic {
					t.Errorf("Dot() recovered panic = %v, wantPanic %v", r, tt.wantPanic)
				}
			}()
			if got := Dot(tt.args.m, tt.args.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Dot() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func BenchmarkDot(b *testing.B) {
	type args struct {
		m mat.Matrix
		n mat.Matrix
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "2x2 matrices",
			args: args{
				m: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				n: mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
			},
		},
		{
			name: "100x100 matrices",
			args: args{
				m: mat.NewDense(100, 100, RandomArray(100*100, 100)),
				n: mat.NewDense(100, 100, RandomArray(100*100, 100)),
			},
		},
		{
			name: "1000x1000 matrices",
			args: args{
				m: mat.NewDense(1000, 1000, RandomArray(1000*1000, 1000)),
				n: mat.NewDense(1000, 1000, RandomArray(1000*1000, 1000)),
			},
		},
		{
			name: "10000x1000 & 1000x10000 matrices",
			args: args{
				m: mat.NewDense(10000, 1000, RandomArray(10000*1000, 10000)),
				n: mat.NewDense(1000, 10000, RandomArray(1000*10000, 1000)),
			},
		},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			_ = Dot(tt.args.m, tt.args.n)
		})
	}
}
