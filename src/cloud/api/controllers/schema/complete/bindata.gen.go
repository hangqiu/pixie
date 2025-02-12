// Code generated for package complete by go-bindata DO NOT EDIT. (@generated)
// sources:
// 01_base_schema.graphql
// 02_unauth_schema.graphql
// 03_auth_schema.graphql
package complete

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func bindataRead(data []byte, name string) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("Read %q: %v", name, err)
	}

	var buf bytes.Buffer
	_, err = io.Copy(&buf, gz)
	clErr := gz.Close()

	if err != nil {
		return nil, fmt.Errorf("Read %q: %v", name, err)
	}
	if clErr != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

type asset struct {
	bytes []byte
	info  os.FileInfo
}

type bindataFileInfo struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
}

// Name return file name
func (fi bindataFileInfo) Name() string {
	return fi.name
}

// Size return file size
func (fi bindataFileInfo) Size() int64 {
	return fi.size
}

// Mode return file mode
func (fi bindataFileInfo) Mode() os.FileMode {
	return fi.mode
}

// Mode return file modify time
func (fi bindataFileInfo) ModTime() time.Time {
	return fi.modTime
}

// IsDir return file whether a directory
func (fi bindataFileInfo) IsDir() bool {
	return fi.mode&os.ModeDir != 0
}

// Sys return file is sys mode
func (fi bindataFileInfo) Sys() interface{} {
	return nil
}

var __01_base_schemaGraphql = []byte("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x6c\x8f\xb1\x4e\xc4\x30\x10\x44\x7b\x7f\xc5\xa0\x14\x54\x5c\x2a\x10\x4a\x49\x4f\x81\xe0\x07\x1c\x7b\x38\x47\x72\xbc\x3e\xef\x46\x47\x84\xf8\x77\x94\xcb\x5d\x77\xd5\x6c\x31\xf3\xb4\x4f\x43\xe2\xec\xf1\xeb\x80\xd3\xc2\xb6\x0e\xf8\xd8\xc2\x01\xf3\x62\xde\x26\x29\x03\xde\xaf\x97\xfb\x73\xae\xc3\x57\x22\xb4\x32\x20\x0a\xb5\x3c\x1a\x7c\xce\x72\x06\xe7\x6a\x2b\x6c\xad\xd4\x83\xeb\xf0\x29\x38\x13\xa1\xd1\x1b\x51\x7d\x0e\x4c\x92\x23\x9b\x22\xb1\x11\xbe\xc4\xeb\xce\x12\x95\xfb\x0e\x26\x18\xe9\x3a\xf0\xc7\x58\x22\x23\xc6\x15\x62\x89\x0d\xdf\x53\xde\xb9\xc9\xac\xea\xd0\xf7\xc7\xc9\xd2\x32\x1e\x82\xcc\xfd\xb1\xf9\x9a\x4e\xf9\x96\x4f\xdb\x73\xfd\xa4\xba\x50\xfb\xe7\x97\x57\xe7\x36\xf8\xae\x75\xf1\x2c\x22\x75\xc0\x9b\x48\xa6\x2f\x0f\x9b\xd4\xa5\x70\xb3\xbc\xdf\xf9\x0f\x00\x00\xff\xff\x6f\xc4\xb8\xef\x28\x01\x00\x00")

func _01_base_schemaGraphqlBytes() ([]byte, error) {
	return bindataRead(
		__01_base_schemaGraphql,
		"01_base_schema.graphql",
	)
}

func _01_base_schemaGraphql() (*asset, error) {
	bytes, err := _01_base_schemaGraphqlBytes()
	if err != nil {
		return nil, err
	}

	info := bindataFileInfo{name: "01_base_schema.graphql", size: 296, mode: os.FileMode(436), modTime: time.Unix(1, 0)}
	a := &asset{bytes: bytes, info: info}
	return a, nil
}

var __02_unauth_schemaGraphql = []byte("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x64\x8d\x31\x0a\x02\x31\x10\x45\xfb\x9c\xe2\x6f\xa7\x57\x48\x67\x23\x58\x28\x88\xa5\x58\x0c\xeb\x6c\x0c\x6c\x26\x4b\x66\x14\x17\xf1\xee\x62\x20\xa2\xd8\x0d\x6f\x1e\xef\xf3\xdd\x58\xce\xb0\x79\x62\xec\xaf\x5c\x66\x3c\x1c\x40\xc5\xe2\x40\xbd\xe9\xa2\x5d\x3b\x4a\xec\x71\xb0\x12\x25\x74\x4b\x8f\x55\x33\x36\x32\xe4\xce\x3d\x9d\xab\x89\x1f\x5c\x53\xd1\x38\xa9\xc7\xb1\x7d\xba\xd3\xbf\x5d\xc5\x1b\x17\x8d\x59\x3e\x23\x0e\xe8\x2f\x24\x81\xc7\x1c\xbe\xa1\xc5\xc4\x6a\x94\xa6\xad\x7a\xac\xc7\x4c\xf6\x0e\xbe\x02\x00\x00\xff\xff\xa4\xc1\x10\x47\xc8\x00\x00\x00")

func _02_unauth_schemaGraphqlBytes() ([]byte, error) {
	return bindataRead(
		__02_unauth_schemaGraphql,
		"02_unauth_schema.graphql",
	)
}

func _02_unauth_schemaGraphql() (*asset, error) {
	bytes, err := _02_unauth_schemaGraphqlBytes()
	if err != nil {
		return nil, err
	}

	info := bindataFileInfo{name: "02_unauth_schema.graphql", size: 200, mode: os.FileMode(436), modTime: time.Unix(1, 0)}
	a := &asset{bytes: bytes, info: info}
	return a, nil
}

var __03_auth_schemaGraphql = []byte("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xac\x59\xdd\x6f\xe3\xb8\x11\x7f\xf7\x5f\x31\x7b\xfb\x70\x0e\x10\x2c\x0e\x45\xef\x50\xf8\xa9\x3a\x5b\x7b\xab\x26\x71\xdc\xd8\xd9\xed\xa1\x58\x2c\x68\x71\x6c\x11\x96\x48\x1d\x49\x39\x71\x8b\xfd\xdf\x8b\x21\xa9\x0f\xda\xca\x6d\x73\xed\x9b\xc5\x8f\x99\xdf\x7c\xf0\x37\x43\x1a\x9f\x2d\x4a\x0e\xf6\x54\x23\xfc\xbd\x41\x7d\x82\x7f\x4f\x00\x1a\x83\x7a\x06\x8f\x06\x75\x26\x77\xea\xcd\x04\x40\xe9\xfd\x0c\xee\xf5\xbe\xfd\xa6\x15\x6b\xb4\x56\xc8\xbd\xf1\x2b\xdb\xaf\x76\x36\xb1\x56\x8b\x6d\x63\x31\xcc\xf7\xdf\x41\x1e\x0d\x9a\x19\xfc\xb3\x53\xf3\x99\x26\xf2\xb2\x31\x16\xf5\x54\xf0\x19\x64\x8b\x37\x57\x33\x98\xfb\x91\x56\x73\x58\xf0\xf3\x69\xc9\x2a\x9c\x4a\x56\xe1\x0c\xd6\x56\x0b\xb9\x7f\x79\x31\xa9\x19\xce\x0c\x35\xcd\x95\x94\x98\x5b\xa1\xe4\xa5\xce\x7e\xae\x17\x28\x12\x6d\xc5\x8e\xe5\x76\xca\xc2\x8f\xcd\xa9\xc6\x19\x24\x83\x2f\x27\xe2\x36\x6b\x87\x68\x23\x6b\xac\xca\x55\x55\x97\x68\x71\x2a\x64\xdd\xd8\x16\xf6\x35\xe4\x8d\x36\x4a\xaf\x94\x99\x41\x26\xed\x35\x30\xa7\x72\x06\xc9\x60\x4f\xe2\xc6\x48\xf8\x75\x8b\xfc\x31\x5b\xb4\x32\xae\xe2\xc5\x0f\x68\x9a\xf2\x42\xed\x7b\x81\x25\x3f\xd7\xbd\xa3\xc1\x60\xc1\x60\x6d\x2a\xad\xb0\xa7\x1b\x21\xf9\xf5\x04\x00\x40\xe3\x6f\x8d\xd0\xc8\x13\xbd\xa7\xc5\xe4\xd0\xf1\xe5\x9f\x5f\x80\x17\x2d\x5f\x37\xfb\x3d\x1a\x32\xe8\xf3\x64\x02\xf0\x16\xd6\xb9\x16\xb5\xad\xf6\x1a\x50\xf2\x5a\x09\x69\xcd\x35\x68\xdc\xa1\x06\xab\x80\xab\xdc\x80\x90\x90\x97\xaa\xe1\xac\x16\xef\x6a\xad\xac\x9a\x00\x94\xe2\x88\x1f\x05\x3e\x11\x9c\xdb\xf0\xfb\x0e\x2d\xe3\xcc\x32\x1f\xe4\x76\xc5\x5c\x49\x8b\xd2\x9a\x41\x8c\x6f\xcf\xa6\x68\xb9\x71\x38\x48\x9c\x47\x14\x0b\xf3\xb3\x23\xa2\xd6\xd1\xc4\x1b\x6f\xd3\x02\xeb\x52\x9d\xe0\x80\x27\x33\x01\xe0\xee\xab\x42\x69\x6f\xf0\x44\x0a\x16\xc3\x81\x58\x4f\xb4\x76\xa0\x26\xda\x12\xb4\x24\xab\xac\x55\xc1\x6a\x11\x64\x27\xab\xec\x42\xa8\x9f\x1d\x48\xf3\x8b\xde\x4c\xbe\x4e\x26\x43\x16\xb8\x6b\x2c\xa3\xc8\x38\x22\x98\x6b\x64\x16\xc3\x69\x88\x4e\x17\xfc\x95\x63\xad\x31\x67\x16\xf9\x54\x23\x33\x94\xb0\xdf\x85\x05\x06\x98\x46\x90\xea\x09\x72\x27\x80\xc3\x51\x30\xa8\x9f\x83\x65\xdf\x5d\x4d\x00\x1e\x6b\xce\x2c\x7e\x14\xff\x12\xee\x9c\xed\xc4\x7e\x1a\x12\x87\xf2\x26\x5b\xbc\xb9\x86\xe3\x60\x72\x06\x29\x17\x96\x6d\xcb\x68\xcb\xc8\x91\xf7\x90\x23\x57\x5d\x78\x0e\x60\x81\x94\x87\x8b\x17\x1c\xfd\xb3\x52\x25\x32\xd9\x8b\xf3\xbe\xea\x7d\xd6\x0a\xf0\xdf\xe3\x3b\xbd\x81\x43\x6a\x9c\x9a\x8e\x31\x5b\x63\x22\xe6\xbc\xba\x64\xd2\x35\xda\x98\x3c\xa7\x6c\xc0\xab\x43\x29\x03\x7e\xbd\x1a\x63\xdc\x4c\x1e\x85\x87\x33\xc5\x8a\x89\xb2\x63\x4d\xe2\x00\x6d\xec\x72\xc8\xa4\xd7\x50\xb2\xb3\xa1\xab\xb6\x20\x90\x98\xd8\xbe\x15\xea\x4a\x18\x23\x94\x34\x53\xa2\xfe\x2e\x80\x4d\x3c\x19\x03\x1e\x4c\xf4\xc2\x87\x31\xbc\xd7\xfb\xa9\xd2\xfb\x73\x14\xd9\xa2\xd7\x7e\xaf\xf7\x9d\x73\x95\xde\x77\x8a\x55\x3f\xde\x2b\x1d\x2c\x26\x39\x5d\x35\xfb\x3a\x99\xb8\xcc\x6f\x11\xb8\xcc\x0f\x21\x9d\x00\x44\x25\x66\x02\x10\x7b\x6f\x02\x50\x8b\xdc\x36\x3a\x5a\x73\x0e\xdb\x0f\xf5\x84\x48\x03\xc2\x24\x75\xad\xd5\x11\xf9\x20\x6d\x5a\x2c\xd9\x22\x5d\x31\x5b\x38\x28\xd9\x22\x3d\x17\x56\x33\x5b\xf4\xdf\xed\xa6\x60\xd1\xb7\xf0\x4b\x72\x86\xd7\xcd\x4a\x13\xe5\xac\xe0\x48\x6a\x89\x43\x02\x02\x22\x8f\xa1\x83\x5a\x0f\x3a\x25\x4c\xb2\xf2\x64\x45\x6e\xee\x6b\xab\xa8\xaa\x44\xa2\x3c\x80\xe1\xe6\x3e\x23\xdd\x76\xab\x1a\xbd\x46\x94\x2f\xed\x73\xa5\xea\x85\x24\x1f\x17\x30\xbe\xeb\xbf\xc2\xdc\x01\x8d\xc9\xf3\xcc\x99\x81\xcf\x12\x7b\x67\x66\xf0\xbe\x54\xcc\x7a\xc2\x36\xf9\x65\x38\xbc\xa0\x33\x01\x07\xa2\x91\x3e\x18\xaf\x91\x37\x5a\x31\xfe\x07\x7c\x91\xbc\xff\x0b\x4c\x94\x4d\x35\xd2\x46\xac\x2d\xb3\xe8\x14\x24\xe9\xfa\xcb\xe3\xf2\x66\x79\xff\x69\x19\xbe\x56\xe9\x72\x91\x2d\x7f\x09\x5f\x0f\x8f\xcb\x65\xff\xf5\x3e\xc9\x6e\xd3\x45\xf8\xd8\xa4\x0f\x77\xd9\x32\xd9\xa4\x8b\x51\x4d\x7d\x7f\xe4\x15\x25\x9b\x81\xa2\xb7\x90\x48\x40\x2e\x6c\x68\xad\x40\xe5\xd4\x73\x81\xd8\x01\x73\x24\x05\x05\x33\x50\x29\x2e\x76\x02\x39\xd8\x02\xc1\x67\x91\xc5\x67\x0b\xdb\x13\x08\x69\x50\x53\x0e\x81\xd2\xc0\x89\xfa\xe9\x77\x5e\x30\xcd\x72\xaa\x77\xef\x9c\x92\x4d\x21\xa8\x4f\xc9\xcb\x86\xa3\xa1\x6a\xea\x36\x48\x27\xef\x80\xa7\xad\x62\x9a\x03\x93\x1c\x6a\x66\xbc\x00\x55\x55\x4c\x72\xb7\x9d\x10\xa7\x8b\x6c\xe3\xe1\x82\xc1\x12\xf3\x1e\xaf\x2c\x4f\xe3\xa0\xf3\x42\x19\x94\xc0\x64\xd4\xea\x81\xe9\x3a\xac\x77\x2d\x2c\x2e\xa8\x58\x1b\x70\x9d\xd3\x5b\x07\x2a\xda\x62\x0b\x66\x41\x58\x30\x85\x6a\x4a\x0e\x95\x3a\xa2\x5b\x44\xaa\xbe\x37\xa1\x49\xa5\x76\x8c\x06\x25\x39\x86\x11\x87\xd4\x5a\x50\x74\x2d\xdb\xb6\x56\xac\xd3\xdb\x74\xbe\xf9\x9d\x7c\xa0\x3e\x31\xa4\xc3\x4d\x94\x0e\x37\x5f\x56\xf7\x8b\xf0\x6b\xfd\x71\xde\xfe\x9a\x3f\x64\xab\x4d\xf8\x58\x26\x77\xe9\x7a\x95\xcc\xd3\xfe\x98\x8d\x36\x96\x4e\xfe\x41\x48\xfe\x52\x5f\x7b\xc6\x8c\x21\x9d\xa9\x8f\x73\xbd\x77\x37\x5a\x31\x9b\x17\xc8\x33\xc9\xf1\xd9\xf5\xbd\x99\xb4\x9f\xa9\x19\xa4\xa4\x1e\x13\xee\xb2\xbd\x43\xb7\x61\xdb\x33\x50\x94\x27\x94\x5f\x1c\x9f\x41\xed\x9c\x37\x2d\xdb\x7a\xf7\xdb\x02\xcd\x30\x78\xbe\x91\xda\x29\x4d\xbe\xb5\x6c\xeb\x50\xb8\x5b\x82\x13\xf4\xa9\x40\x5b\xa0\x0e\xc9\x42\x19\xc5\x06\x9b\x69\x1f\x58\x0a\x3e\xc9\xf7\x0a\x9f\x44\x59\x42\xc5\x0e\x3e\xb4\x21\xff\x00\x9f\x31\x6f\x1c\x5d\x92\x9e\xfe\x2b\xd9\x59\x62\x4f\x12\xde\xf3\x24\x0c\xf1\xfd\x4e\x63\x3f\x16\x1f\x7f\x31\x19\xb8\x61\xa7\x74\xc5\x2c\x75\x88\xfe\xc0\x11\xd8\xee\xf4\x99\x70\x47\x79\x2a\x44\x5e\xb8\x6c\xdf\x22\x4a\xa8\x99\x36\xc8\xe9\x58\x5e\xe6\xb0\xea\x12\xdd\x27\x39\xdb\xae\xad\xaa\xa1\x56\x46\x38\xbc\x64\x5f\xa7\x33\x1b\x5e\x85\x22\x87\x9e\x63\x20\x5c\x0c\x8e\xac\x14\xfc\x7a\xe0\x9f\xd6\x81\xef\x5c\x39\x4f\xbb\xf1\xa1\xb3\xde\x42\x52\x96\x51\x48\x29\x2c\xc8\xf2\x62\x10\x7d\x02\x69\x42\x8c\xd7\x91\x77\xa3\xfc\xe9\x9d\x4a\x77\x0d\x26\x24\x6a\xca\xb6\xc6\x57\xb6\xf3\x42\x3f\x4e\xda\x21\x6f\xfb\x65\x15\x1a\xc3\xf6\xd1\x50\xdb\xcf\x0f\x47\x8c\x65\xda\xce\x55\x23\xad\xcb\xbf\xbe\x8c\xdc\xfc\xc5\xa4\x47\x94\x3e\xaa\x23\xc2\x5c\x77\xb9\x11\x15\x46\x30\xa8\xbf\x3c\x1b\x6c\x05\xae\x14\xff\x43\x56\x35\xe6\xd5\x66\xe5\xad\x1b\xdd\x0b\x41\xec\x53\x7f\x6d\x42\x32\x8d\x66\x5b\x33\xfd\xf0\xb8\x3f\x1c\xdf\x85\xfb\xc8\xc0\x04\x9f\xea\x1c\x77\x8c\x92\xdf\x05\x80\x58\x5c\x2a\x5b\x84\xdc\x3a\x48\xf5\x24\x29\xfe\xf3\x75\x54\xb6\x68\x5f\x58\x6f\xa0\x40\x56\xda\xe2\x44\x5b\x0b\x64\xda\x6e\x91\x59\x4f\x10\x1a\x73\x14\x47\xe4\x54\x6c\x34\xee\x9b\x92\x69\x10\xd2\xa2\xa6\x06\xcf\x55\x1c\x5b\xf8\x03\x11\xae\x58\x24\x4e\xa3\xa9\x95\xe4\x84\xc0\x2a\x77\xc1\x47\x63\x4d\x00\xf1\x21\x4d\x6e\x37\x1f\x7e\xbd\x04\xd1\xc8\x01\x0c\xc7\x21\xbd\xc4\xdc\x3f\x97\x50\x05\x55\xb0\x12\xcf\x02\x61\x4e\x57\x76\x87\x40\x18\xa0\x8e\x53\xf0\xf6\xac\xf5\x36\x5c\xc3\xd6\x1d\x7d\xf9\xbd\x85\xdf\x1a\xd4\x27\x77\xb6\xe8\x98\x18\x55\x61\x08\x5b\xa8\x63\x1a\x0d\x56\xdb\x12\x0d\x7c\xd8\x6c\x56\xdf\x1b\xf8\xf1\x87\x1f\x42\xf4\x3b\xff\x8d\x83\x77\xd4\xb7\x57\xee\x41\x41\x98\x1e\x6b\xb0\xe3\x97\x87\xd5\xbc\xb5\x80\xc8\x73\xab\x91\x1d\xcc\x3b\x27\xa0\x50\x35\x7a\x6a\x62\xb6\x2b\x9e\xad\xe1\x4e\x6e\x4e\x40\xb7\x2c\x3f\x50\xa9\x16\x12\x9d\xc9\x1a\x4d\x53\x11\x91\x40\x40\xe4\x91\x04\x9c\x8b\x6c\x3d\xbf\x5f\x2e\xd3\xf9\xc6\xf5\x38\xe7\x7e\xa6\xfb\x0d\xc5\xe6\xa9\x40\x79\xee\x68\xe1\x47\x6a\xad\x72\x34\x86\x78\xa4\x5d\xde\xfa\x60\xb5\x48\x36\xbe\x91\xf2\x72\xfd\x55\xda\x77\x0c\xad\xe5\xde\xed\x34\x24\x95\x05\x43\x47\x98\xc9\x13\x28\xc7\x80\xbb\x46\xfb\xd2\xe2\xd3\xd8\xc9\x47\x03\x6c\xab\x1a\xef\x82\xa7\x40\x95\xc2\x0e\x73\x53\xe9\x73\x28\x97\x36\x06\x2c\x4f\xcc\x80\xd5\xa7\x90\x7f\x5e\x81\x87\xb4\x63\xa2\xc4\x2e\x6b\xa4\x7a\x22\x83\x19\x6c\x19\x8f\x1c\xe8\x8c\x4c\xdb\x2e\xb1\x65\x8f\xe1\xf3\x80\x3b\x7d\x35\x33\xc6\x16\x5a\x35\xfb\x22\x75\x57\x9f\xb1\xfb\xd6\xf0\x65\x23\xee\x84\x5b\x66\x89\x8e\x75\xcb\x60\x1f\xda\x1c\x8e\xc8\x28\x7e\xb7\x88\xde\x2b\xba\xd9\x8f\xa8\x8d\x38\x23\x23\xaf\xe1\xe5\x99\x8b\x5b\xa0\x46\x6b\x4f\xf3\xf1\xc9\xcb\x57\xb8\x96\xf0\xb4\x2a\x57\x25\x93\xd8\xf1\xac\x6b\x6b\xba\x2f\x4f\x70\xdd\x39\x5f\x30\xcb\xbe\xbd\x5c\x36\xd5\x52\x71\x34\x81\x0b\xdd\x40\x26\x8d\xd5\x0d\xdd\x2f\x90\xc7\x93\xde\xa7\x77\x97\x0c\x5d\x6b\x3c\x0a\xd5\x98\xf5\x98\xd3\x2f\xe6\xa3\xfa\x71\x1e\xca\xf8\xed\xd6\x07\xb5\x4e\x38\xd7\x68\xa2\x3a\x61\xd5\x01\xe5\xe5\xe5\xa8\x7f\xeb\x70\x5b\x2f\x2e\xfd\xc2\xcd\xdd\x0a\x79\x88\xf6\xbe\x85\x87\x6f\xbc\x5a\x3a\xe9\xe7\x8f\x95\xdf\xb8\xb2\x5f\x5c\xb4\x5e\xa9\xa6\x7d\x99\x0c\x25\xda\xeb\x9c\x5d\xa0\x70\x11\x78\x2e\xdb\xd5\x43\x04\x47\x61\xfe\xb6\xbe\x5f\xfe\x11\x10\xf1\x4b\xea\xab\x2c\x05\x62\xa7\x16\x65\x7c\x6a\x5f\xa5\xfc\x05\xfb\xcf\xde\x78\xc3\xf1\x88\x4d\xef\x6e\x31\x83\xe7\x7d\x27\x06\x20\xba\x62\xba\xcf\xdb\x6c\xf9\xf8\x8f\x2f\xc9\xdd\xe2\xa7\x3f\xb7\x43\x8b\xe4\xe1\x53\xb6\x8c\xc7\xe6\xf7\xcb\x4d\x92\x2d\xd3\x87\x2f\xeb\x74\xf3\xe5\xd7\xe4\xee\x76\x3d\x3e\x35\x22\x2f\x5e\xb0\x49\xef\x56\xb7\x44\x82\x5e\x48\x77\x04\xfa\xff\x1e\xfc\xff\x39\x3a\xca\x5d\x53\xb0\x3f\xfd\xf8\x53\x64\x63\xfc\x68\xf2\x1a\x0e\x1d\x7f\x72\x19\x3c\xee\xf9\x88\x5f\x3e\x76\x5d\x6e\x1c\x3c\xd0\xf9\x43\xf7\xc2\x4b\xd5\xe4\xeb\xe4\x3f\x01\x00\x00\xff\xff\x39\x1f\x66\x63\xb8\x1a\x00\x00")

func _03_auth_schemaGraphqlBytes() ([]byte, error) {
	return bindataRead(
		__03_auth_schemaGraphql,
		"03_auth_schema.graphql",
	)
}

func _03_auth_schemaGraphql() (*asset, error) {
	bytes, err := _03_auth_schemaGraphqlBytes()
	if err != nil {
		return nil, err
	}

	info := bindataFileInfo{name: "03_auth_schema.graphql", size: 6840, mode: os.FileMode(436), modTime: time.Unix(1, 0)}
	a := &asset{bytes: bytes, info: info}
	return a, nil
}

// Asset loads and returns the asset for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func Asset(name string) ([]byte, error) {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[cannonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("Asset %s can't read by error: %v", name, err)
		}
		return a.bytes, nil
	}
	return nil, fmt.Errorf("Asset %s not found", name)
}

// MustAsset is like Asset but panics when Asset would return an error.
// It simplifies safe initialization of global variables.
func MustAsset(name string) []byte {
	a, err := Asset(name)
	if err != nil {
		panic("asset: Asset(" + name + "): " + err.Error())
	}

	return a
}

// AssetInfo loads and returns the asset info for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func AssetInfo(name string) (os.FileInfo, error) {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[cannonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("AssetInfo %s can't read by error: %v", name, err)
		}
		return a.info, nil
	}
	return nil, fmt.Errorf("AssetInfo %s not found", name)
}

// AssetNames returns the names of the assets.
func AssetNames() []string {
	names := make([]string, 0, len(_bindata))
	for name := range _bindata {
		names = append(names, name)
	}
	return names
}

// _bindata is a table, holding each asset generator, mapped to its name.
var _bindata = map[string]func() (*asset, error){
	"01_base_schema.graphql":   _01_base_schemaGraphql,
	"02_unauth_schema.graphql": _02_unauth_schemaGraphql,
	"03_auth_schema.graphql":   _03_auth_schemaGraphql,
}

// AssetDir returns the file names below a certain
// directory embedded in the file by go-bindata.
// For example if you run go-bindata on data/... and data contains the
// following hierarchy:
//     data/
//       foo.txt
//       img/
//         a.png
//         b.png
// then AssetDir("data") would return []string{"foo.txt", "img"}
// AssetDir("data/img") would return []string{"a.png", "b.png"}
// AssetDir("foo.txt") and AssetDir("notexist") would return an error
// AssetDir("") will return []string{"data"}.
func AssetDir(name string) ([]string, error) {
	node := _bintree
	if len(name) != 0 {
		cannonicalName := strings.Replace(name, "\\", "/", -1)
		pathList := strings.Split(cannonicalName, "/")
		for _, p := range pathList {
			node = node.Children[p]
			if node == nil {
				return nil, fmt.Errorf("Asset %s not found", name)
			}
		}
	}
	if node.Func != nil {
		return nil, fmt.Errorf("Asset %s not found", name)
	}
	rv := make([]string, 0, len(node.Children))
	for childName := range node.Children {
		rv = append(rv, childName)
	}
	return rv, nil
}

type bintree struct {
	Func     func() (*asset, error)
	Children map[string]*bintree
}

var _bintree = &bintree{nil, map[string]*bintree{
	"01_base_schema.graphql":   &bintree{_01_base_schemaGraphql, map[string]*bintree{}},
	"02_unauth_schema.graphql": &bintree{_02_unauth_schemaGraphql, map[string]*bintree{}},
	"03_auth_schema.graphql":   &bintree{_03_auth_schemaGraphql, map[string]*bintree{}},
}}

// RestoreAsset restores an asset under the given directory
func RestoreAsset(dir, name string) error {
	data, err := Asset(name)
	if err != nil {
		return err
	}
	info, err := AssetInfo(name)
	if err != nil {
		return err
	}
	err = os.MkdirAll(_filePath(dir, filepath.Dir(name)), os.FileMode(0755))
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(_filePath(dir, name), data, info.Mode())
	if err != nil {
		return err
	}
	err = os.Chtimes(_filePath(dir, name), info.ModTime(), info.ModTime())
	if err != nil {
		return err
	}
	return nil
}

// RestoreAssets restores an asset under the given directory recursively
func RestoreAssets(dir, name string) error {
	children, err := AssetDir(name)
	// File
	if err != nil {
		return RestoreAsset(dir, name)
	}
	// Dir
	for _, child := range children {
		err = RestoreAssets(dir, filepath.Join(name, child))
		if err != nil {
			return err
		}
	}
	return nil
}

func _filePath(dir, name string) string {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	return filepath.Join(append([]string{dir}, strings.Split(cannonicalName, "/")...)...)
}
