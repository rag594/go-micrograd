package main

import (
	"fmt"
)

/*
*

q = x + y
f = q * z

df/dq = z ---- intermediate

df/dq = z * dq/dq

dq/dx = 1.0
dq/dy = 1.0

df/dz = q
df/dx = df/dq * dq/dx = z * 1.0
df/dy = df/dq * dq/dy = z * 1.0

	x
	 \
	   + -> q
	 /       \
	y          * -> f
	         /
	        z

x, y, q, z, f
*/

func main() {
	//h := 0.01

	x := ScalarValue(-2.0)
	y := ScalarValue(5.0)

	q := x.Add(y)

	z := ScalarValue(-4)

	// forward pass
	f := q.Mul(z)

	// backward pass
	f.Grad = 1.0

	backwardPassGraph := NewBackwardPassGraph()

	f.backward(backwardPassGraph)

	fmt.Printf("%+v\n", f)
}
