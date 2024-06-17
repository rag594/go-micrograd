# Go-Micrograd

### Backward Pass Graph of the expression - Backpropogation

![image](assets/outimage.png)

#### To run the example of backpropogation

```console
foo@bar:~$ cd examples/backpropogation
foo@bar:~$ go run .
```

**_NOTE:_**  The image is created as `outimage.png` under `examples/backpropogation` package

### Multi Layer Perceptron - With expression graph

Below is a MLP(3,4,4,1) of which expression graph is visualized

<img src="https://github.com/rag594/go-micrograd/assets/8286518/f2db8b62-9730-4979-ad6b-fd49a5bcb8a2" width="350" height="350">

Expression graph of above MLP

<img src="https://github.com/rag594/go-micrograd/assets/8286518/779b18d0-92e6-470d-b333-c6b71a750160">


#### To run the example of backpropogation

```console
foo@bar:~$ cd examples/mlp
foo@bar:~$ go run .
```

**_NOTE:_**  The image is created as `outimage.png` under `examples/mlp` package

##### inspired from https://github.com/karpathy/micrograd
