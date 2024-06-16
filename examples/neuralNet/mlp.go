package main

import (
	"fmt"
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

	d := l.MLPOutut(x)
	d.Label = "g"

	f := d.Tanh()
	f.Label = "final"

	f.Grad = 1.0

	fmt.Println(f)

	// initialise the tracer
	t, err := tracer.NewTracer()

	if err != nil {
		log.Fatalln(err)
	}

	// Initialize a new backward pass graph
	backwardPassGraph := core.NewBackwardPassGraph()

	f.Backward(backwardPassGraph)

	t.Draw(f)
}
