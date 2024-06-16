package tracer

import (
	"bytes"
	"fmt"
	"image/png"
	"log"
	"os"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"github.com/rag594/go-micrograd/core"
)

// Trace represents the state of graph of expression to be traced/drawn
type Trace struct {
	Graph   *cgraph.Graph
	Viz     *graphviz.Graphviz
	Visited map[*core.Value]bool
}

// NewTracer initialises the Trace
func NewTracer() (*Trace, error) {
	g := graphviz.New()
	graph, err := g.Graph(graphviz.Directed)
	if err != nil {
		fmt.Println("Err ", err)
		return nil, err
	}

	graph.SetRankDir(cgraph.LRRank)

	return &Trace{Graph: graph, Viz: g, Visited: make(map[*core.Value]bool)}, nil
}

// createGraph traverses the graph in dfs fashion and creates visual nodes
func (t *Trace) createGraph(root *core.Value) {
	if !t.Visited[root] {
		t.Visited[root] = true
		var r, op *cgraph.Node
		r, _ = t.Graph.CreateNode(root.Label)
		r.SetLabel(fmt.Sprintf("| data %.2f | grad %.2f |", root.Data, root.Grad))
		r.SetXLabel(root.Label)
		if root.Op != "" {
			op, _ = t.Graph.CreateNode(root.Label + root.Op)
			op.SetLabel(root.Op)
			_, _ = t.Graph.CreateEdge("", op, r)
		}
		for _, child := range root.Children {
			c, _ := t.Graph.CreateNode(child.Label)
			c.SetLabel(fmt.Sprintf("| data %.2f | grad %.2f |", child.Data, child.Grad))
			c.SetXLabel(child.Label)
			if op != nil {
				_, _ = t.Graph.CreateEdge("", c, op)
			} else {
				_, _ = t.Graph.CreateEdge("", c, r)
			}
			t.createGraph(child)
		}
	}
}

// Draw render the graph's image in a file
func (t *Trace) Draw(v *core.Value) {
	t.createGraph(v)
	var buf bytes.Buffer
	if err := t.Viz.Render(t.Graph, graphviz.PNG, &buf); err != nil {
		log.Fatal(err)
	}

	image, err := t.Viz.RenderImage(t.Graph)
	if err != nil {
		log.Fatal(err)
	}

	f, err := os.Create("outimage.png")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	err = png.Encode(f, image)
	if err != nil {
		log.Fatal(err)
	}

}
