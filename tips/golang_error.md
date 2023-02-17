## Golang Error

Using reference to loop iterator variable

```golang
in := []int{1, 2, 3}

var out []*int
for  _, v := range in {
    out = append(out, &v)
}

fmt.Println("Values:", *out[0], *out[1], *out[2])
fmt.Println("Addresses:", out[0], out[1], out[2])
```

Bad ordered struct

```golang
type BadOrderedPerson struct {
    Veteran bool   // 1 byte
    Name    string // 16 byte
    Age     int32  // 4 byte
}

type OrderedPerson struct {
    Name    string
    Age     int32
    Veteran bool
}
```

```golang
type BadOrderedPerson struct {
    Veteran bool     // 1 byte
    _       [7]byte  // 7 byte: padding for alignment
    Name    string   // 16 byte
    Age     int32    // 4 byte
    _       struct{} // to prevent unkeyed literals
    // zero sized values, like struct{} and [0]byte occurring at
    // the end of a structure are assumed to have a size of one byte.
    // so padding also will be addedd here as well.

}

type OrderedPerson struct {
    Name    string
    Age     int32
    Veteran bool
    _       struct{}
}
```
