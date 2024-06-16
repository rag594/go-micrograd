package core

import (
	"math"
	"slices"
)

const (
	add   = "+"
	mul   = "*"
	tanhC = "tanh"
)

// Value encapsulates the data and the corresponding nodes in the expression
// A simple example can be q = 1 or can be a complex expression f = a*b
type Value struct {
	Data         float64  // data associated with Value
	Children     []*Value // child nodes of an expression or with which expression is build up
	Grad         float64  // gradient of the expression
	Op           string   // operation involved in the expression may or may not be empty
	BackwardFunc func()   // backpropogate func to calculate the gradient/derivative
	Label        string
}

// backward backpropogates to the expression to calculate the change/gradient of each node
func (operandA *Value) Backward(backwardPassGraph *BackwardPassGraph) {
	backwardPassGraph.buildBackwardPassOrder(operandA)
	slices.Reverse(*backwardPassGraph.deps)
	for _, val := range *backwardPassGraph.deps {
		if val.BackwardFunc != nil {
			val.BackwardFunc()
		}
	}
}

// NewValue initialises the expression with op involved
func NewValue(data float64, children []*Value, op string) *Value {
	return &Value{
		Data:     data,
		Children: children,
		Grad:     0.0,
		Op:       op,
	}
}

// ScalarValue initialises the scalar value
func ScalarValue(data float64, label string) *Value {
	return &Value{
		Data:  data,
		Grad:  0.0,
		Label: label,
	}
}

// Add is the ops used in the expression
func (operandA *Value) Add(operandB *Value) *Value {
	out := NewValue(operandA.Data+operandB.Data, []*Value{operandA, operandB}, add)
	backwardFunc := func() {
		operandA.Grad += out.Grad
		operandB.Grad += out.Grad
	}
	out.BackwardFunc = backwardFunc
	return out
}

// Mul is the ops used in the expression
func (operandA *Value) Mul(operandB *Value) *Value {
	out := NewValue(operandA.Data*operandB.Data, []*Value{operandA, operandB}, mul)
	backwardFunc := func() {
		operandA.Grad += operandB.Data * out.Grad
		operandB.Grad += operandA.Data * out.Grad
	}
	out.BackwardFunc = backwardFunc
	return out
}

// tanh is the ops used in the expression
func (operandA *Value) Tanh() *Value {
	x := operandA.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := NewValue(t, []*Value{operandA}, tanhC)
	backwardFunc := func() {
		operandA.Grad += (1 - math.Pow(t, 2)) * out.Grad
	}
	out.BackwardFunc = backwardFunc
	return out
}
