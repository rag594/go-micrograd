package main

import (
	"log"

	"github.com/rag594/go-micrograd/core"
	"github.com/rag594/go-micrograd/mlp"
	"github.com/rag594/go-micrograd/tracer"
)

func main() {
	// x = [2,3,-1]
	x := []*core.Value{core.ScalarValue(2.0, "x0"), core.ScalarValue(3.0, "x1"), core.ScalarValue(-1.0, "x2")}

	// [3,4,4,1]
	l := mlp.NewMLP(3, []int{4, 4, 1})

	// forward pass
	d := l.Output(x)
	d.Label = "g"

	// apply activation
	f := d.Tanh()
	f.Label = "final"

	// Initialize a new backward pass graph
	backwardPassGraph := core.NewBackwardPassGraph()

	// backward pass - back propogation
	f.Backward(backwardPassGraph)

	// initialise the tracer
	t, err := tracer.NewTracer()

	if err != nil {
		log.Fatalln(err)
	}

	t.Draw(f)
}
