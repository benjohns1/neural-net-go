package storage

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

// File storage configuration.
type File struct {
	Marshaller
}

// NewJSONFile default JSON storage implementation.
func NewJSONFile() File {
	return File{
		Marshaller: jsonMarshaller{},
	}
}

// Save stores data to disk.
func (f File) Save(v interface{}, path string) error {
	data, err := f.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshaling: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return fmt.Errorf("creating directories: %v", err)
	}
	if err := ioutil.WriteFile(path, data, os.ModePerm); err != nil {
		return fmt.Errorf("writing file: %v", err)
	}
	return nil
}

// Load loads data from disk.
func (f File) Load(v interface{}, path string) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading from file: %v", err)
	}
	if err := f.Unmarshal(data, v); err != nil {
		return fmt.Errorf("unmarshalling: %v", err)
	}
	return nil
}
