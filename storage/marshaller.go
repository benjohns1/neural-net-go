package storage

import "encoding/json"

// Marshaller encodes/decodes interfaces to/from byte arrays.
type Marshaller interface {
	Marshal(v interface{}) ([]byte, error)
	Unmarshal(data []byte, v interface{}) error
}

type jsonMarshaller struct{}

func (j jsonMarshaller) Marshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

func (j jsonMarshaller) Unmarshal(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}
