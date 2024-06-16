package mlp

import (
	"fmt"
	"math/rand"

	"github.com/rag594/go-micrograd/core"
)

// Neuron respresents a single neuron
type Neuron struct {
	Weights    []*core.Value
	Bias       *core.Value
	LayerLabel string
}

// Layer represents the hidden layers in MLP
type Layer struct {
	Neurons    []*Neuron
	LayerLabel string
}

// MLP represents the multi layer NN with layers specified
type MLP struct {
	Layers []*Layer
}

// NewMLP takes the input to the MLP along with output layers
func NewMLP(ninput int, noutputs []int) *MLP {
	noutputs = append([]int{ninput}, noutputs...)
	fmt.Println(noutputs)
	layers := []*Layer{}
	for i := 0; i < len(noutputs)-1; i++ {
		layers = append(layers, NewLayer(noutputs[i], noutputs[i+1], fmt.Sprintf("L%d", i)))
	}
	return &MLP{Layers: layers}
}

// MLPOutut calculates the final output given the input x
func (m *MLP) MLPOutut(x []*core.Value) *core.Value {
	for _, layer := range m.Layers {
		x = layer.Output(x)
	}

	return x[0]
}

// NewLayer initialises the hidden layer with ninput and noutput
func NewLayer(ninput int, noutput int, layerLabel string) *Layer {
	neurons := []*Neuron{}
	for i := 0; i < noutput; i++ {
		neurons = append(neurons, NewNeuron(ninput, layerLabel))
	}
	return &Layer{Neurons: neurons}
}

// Output of the hidden layer
func (l *Layer) Output(x []*core.Value) []*core.Value {
	outs := []*core.Value{}
	for _, n := range l.Neurons {
		outs = append(outs, n.Output(x))
	}

	return outs
}

// NewNeuron initialises the neuron with ninput
func NewNeuron(ninput int, layerLabel string) *Neuron {
	weights := make([]*core.Value, 0)
	for i := 0; i < ninput; i++ {
		weights = append(weights, core.ScalarValue(rand.Float64(), fmt.Sprintf("%sw%d", layerLabel, i)))
	}
	return &Neuron{
		Weights:    weights,
		Bias:       core.ScalarValue(rand.Float64(), fmt.Sprintf("%sb", layerLabel)),
		LayerLabel: layerLabel,
	}
}

// Output output of the neuron
func (n *Neuron) Output(x []*core.Value) *core.Value {
	s := core.ScalarValue(0.0, "")
	products := []*core.Value{}
	for i, d := range x {
		y := d.Mul(n.Weights[i])
		y.Label = fmt.Sprintf("%sy%d", n.LayerLabel, i)
		products = append(products, y)
	}
	for i := 0; i < len(products)-1; i++ {
		z := products[i].Add(products[i+1])
		z.Label = fmt.Sprintf("%sz%d", n.LayerLabel, i)
		s = s.Add(z)
		s.Label = fmt.Sprintf("%sout%d", n.LayerLabel, i)
	}
	k := s.Add(n.Bias)
	k.Label = fmt.Sprintf("%sf", n.LayerLabel)
	return k
}
