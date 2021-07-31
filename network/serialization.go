package network

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func (n *Network) MarshalJSON() ([]byte, error) {
	layers := make([]jsonMatrix, 0, len(n.weights))
	for _, weight := range n.weights {
		layers = append(layers, jsonMatrix{M: weight})
	}
	s := storage{
		Version: 1,
		Config:  n.cfg,
		Layers:  layers,
	}
	return json.Marshal(s)
}

func (n *Network) UnmarshalJSON(data []byte) error {
	var s storage
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	weights := make([]*mat.Dense, 0, len(s.Layers))
	for _, l := range s.Layers {
		weights = append(weights, l.M)
	}
	newNetwork, err := New(s.Config, weights)
	if err != nil {
		return err
	}
	*n = *newNetwork
	return nil
}

type storage struct {
	Version uint32
	Config  Config
	Layers  []jsonMatrix
}

type jsonMatrix struct {
	M *mat.Dense
}

func (m *jsonMatrix) MarshalJSON() ([]byte, error) {
	d, err := m.M.MarshalBinary()
	if err != nil {
		return d, fmt.Errorf("marshaling matrix: %v", err)
	}
	b64 := base64.StdEncoding.EncodeToString(d)
	return []byte(fmt.Sprintf("\"%s\"", b64)), nil
}

func (m *jsonMatrix) UnmarshalJSON(data []byte) error {
	trimmed := strings.Trim(string(data), "\"")
	dec, err := base64.StdEncoding.DecodeString(trimmed)
	if err != nil {
		return fmt.Errorf("base64 decoding: %v", err)
	}
	m.M = new(mat.Dense)
	if err := m.M.UnmarshalBinary(dec); err != nil {
		return fmt.Errorf("unmarshaling matrix: %v", err)
	}
	return nil
}
