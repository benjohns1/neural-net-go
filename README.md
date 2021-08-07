# Neural Net Go
Simple neural network implementation in Go.
## Build
`go build`
## Help
`./neural-net-go -help` to display all available flags.
## Run the MNIST sample
Download the MNIST training and test data from [https://pjreddie.com/projects/mnist-in-csv/](https://pjreddie.com/projects/mnist-in-csv/) and place in a *datasets* directory.
```
./neural-net-go -action=train -epochs=3 
```
```
./neural-net-go -action=test 
```

After training, the model is saved to a JSON file. You can load the same model to train additional epochs or test its accuracy.

## References
Thanks to:
- Daniel Whitenack for [Building a neural network from scratch in Go](https://datadan.io/blog/neural-net-with-go)
- Chang Sau Sheong for [How to build a simple artificial neural network with Go](https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/)
