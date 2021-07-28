package main

import (
	"log"
	"neural-net-go/network"
)

func main() {
	var seed uint64
	//seed = uint64(time.Now().UnixNano())
	nn := network.New(network.Config{
		Input:  3,
		Hidden: 4,
		Output: 2,
		Rate:   0.01,
		Seed:   seed,
	})
	nn.Train([]float64{1,2,3}, []float64{0,1})
	if err := nn.Save(); err != nil {
		log.Fatal(err)
	}
	p := nn.Predict([]float64{1,2,3})
	log.Printf("seed: %d, %+v", seed, p)
}
