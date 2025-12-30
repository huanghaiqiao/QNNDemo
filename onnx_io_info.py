import onnx

model = onnx.load("model/best_sim.onnx")
for input_tensor in model.graph.input:
    print("Input:", input_tensor.name, input_tensor.type)
for output_tensor in model.graph.output:
    print("Output:", output_tensor.name, output_tensor.type)

