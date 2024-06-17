package main

import (
	"log"

	"github.com/rag594/go-micrograd/core"
	"github.com/rag594/go-micrograd/tracer"
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
	x1 := core.ScalarValue(2.0, "x1")
	x2 := core.ScalarValue(0.0, "x2")

	// weights
	w1 := core.ScalarValue(-3.0, "w1")
	w2 := core.ScalarValue(1.0, "w2")

	// bias b
	b := core.ScalarValue(6.8813735870195432, "b")

	// x1w1
	x1w1 := x1.Mul(w1)
	x1w1.Label = "x1w1"

	// x2w2
	x2w2 := x2.Mul(w2)
	x2w2.Label = "x2w2"

	// x1w1 + x2w2
	x1w1x2w2 := x1w1.Add(x2w2)
	x1w1x2w2.Label = "x1w1x2w2"

	// x1w1 + x2w2 + b - forward pass
	n := x1w1x2w2.Add(b)
	n.Label = "n"

	// tanh(x1w1 + x2w2 + b) - apply activation
	out := n.Tanh()
	out.Label = "out"

	// Initialize a new backward pass graph
	backwardPassGraph := core.NewBackwardPassGraph()

	// backpropogate - backward pass
	out.Backward(backwardPassGraph)

	// initialise the tracer
	t, err := tracer.NewTracer()

	if err != nil {
		log.Fatalln(err)
	}

	// Draw the expression graph for tracing
	t.Draw(out)

}
