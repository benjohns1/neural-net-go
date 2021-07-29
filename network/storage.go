package network

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

type storage struct {
	Version uint32 `json:"v"`
	Config Config `json:"c"`
	Layers []jsonMatrix `json:"l"`
}

// SaveFile stores a network to disk.
func (n Network) SaveFile(path string) error {
	s := storage{
		Version: 1,
		Config: n.cfg,
		Layers: []jsonMatrix{
			{M: n.hiddenWeights},
			{M: n.outputWeights},
		},
	}
	data, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("json marshaling: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return fmt.Errorf("creating directories: %v", err)
	}
	if err := ioutil.WriteFile(path, data, os.ModePerm); err != nil {
		return fmt.Errorf("writing file: %v", err)
	}
	return nil
}

// LoadFile loads a network from disk.
func LoadFile(path string) (*Network, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading from file: %v", err)
	}
	var s storage
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("json unmarshaling: %v", err)
	}
	weights := make([]*mat.Dense, 0, len(s.Layers))
	for _, l := range s.Layers {
		weights = append(weights, l.M)
	}
	return New(s.Config, weights), nil
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