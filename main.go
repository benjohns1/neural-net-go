package main

import (
	"log"
	"neural-net-go/network"
)

func main() {
	var seed uint64
	//seed = uint64(time.Now().UnixNano())
	nn := network.NewRandom(network.Config{
		Input:  3,
		Hidden: 5,
		Output: 2,
		Rate:   0.01,
		Seed:   seed,
	})
	nn.Train([]float64{1,2,3}, []float64{0,1})
	p1 := nn.Predict([]float64{1,2,3})
	p2 := nn.Predict([]float64{5,6,7})
	log.Printf("seed: %d:\n\t%+v\n\t%+v", seed, p1, p2)
	if err := nn.SaveFile("data/n1.model"); err != nil {
		log.Fatal(err)
	}
	ln, err := network.LoadFile("data/n1.model")
	if err != nil {
		log.Fatal(err)
	}
	p3 := ln.Predict([]float64{1,2,3})
	p4 := ln.Predict([]float64{5,6,7})
	log.Printf("\n\t%+v\n\t%+v", p3, p4)
}
