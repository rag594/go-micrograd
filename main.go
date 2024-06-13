package main

import (
	"log"
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
	// inputs
	x1 := ScalarValue(2.0, "x1")
	x2 := ScalarValue(0.0, "x2")

	// weights
	w1 := ScalarValue(-3.0, "w1")
	w2 := ScalarValue(1.0, "w2")

	// bias b
	b := ScalarValue(6.8813735870195432, "b")

	// x1w1
	x1w1 := x1.Mul(w1)
	x1w1.Label = "x1w1"

	// x2w2
	x2w2 := x2.Mul(w2)
	x2w2.Label = "x2w2"

	// x1w1 + x2w2
	x1w1x2w2 := x1w1.Add(x2w2)
	x1w1x2w2.Label = "x1w1x2w2"

	// x1w1 + x2w2 + b
	n := x1w1x2w2.Add(b)
	n.Label = "n"

	// tanh(x1w1 + x2w2 + b)
	out := n.tanh()
	out.Label = "out"

	// Initialize a new backward pass graph
	backwardPassGraph := NewBackwardPassGraph()

	// (f(out+h) - f(h) / h) => 1
	out.Grad = 1.0

	// backpropogate
	out.backward(backwardPassGraph)

	// initialise the tracer
	t, err := NewTracer()

	if err != nil {
		log.Fatalln(err)
	}

	// Draw the expression graph for tracing
	t.Draw(out)
}
