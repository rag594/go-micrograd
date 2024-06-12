package main

// BackwardPassGraph maintains the visited and dependency state
type BackwardPassGraph struct {
	visited map[*Value]bool
	deps    *[]*Value
}

// NewBackwardPassGraph initialise the state
func NewBackwardPassGraph() *BackwardPassGraph {

	return &BackwardPassGraph{
		visited: make(map[*Value]bool),
		deps:    new([]*Value),
	}
}

// buildBackwardPassOrder applies the topo sort to build/resolve the expression dependeny
func (b *BackwardPassGraph) buildBackwardPassOrder(root *Value) {
	if !b.visited[root] {
		b.visited[root] = true
		for _, child := range root.Children {
			b.buildBackwardPassOrder(child)
		}
		*b.deps = append(*b.deps, root)
	}
}
