package main

import (
	"log"
)

func main() {
	if err := mnistRun(mnistParseCmdFlags()); err != nil {
		log.Fatalf("mnist error: %v", err)
	}
}
