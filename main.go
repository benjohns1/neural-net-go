package main

import (
	"log"
)

func main() {
	cfg, err := mnistParseCmdFlags()
	if err != nil {
		log.Fatalf("mnist config error: %v", err)
	}
	if err := mnistRun(cfg); err != nil {
		log.Fatalf("mnist error: %v", err)
	}
}
