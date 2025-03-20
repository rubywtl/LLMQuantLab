# LLMQuantLab

This codebase evaluates the efficiency of three quantization methods, ATOM, QServe, and QuaRot, on GPUs by comparing their performance using industry-standard metrics. It focuses on key factors such as prefill tokens, decode length, and batch size, providing insights into how these methods scale on throughput at different stages and their impact on latency. The evaluation is designed to reflect practical industry benchmarks, offering a deeper understanding of the trade-offs in efficiency when applied to large-scale machine learning models.


## Benchmarking Steps
This codebase provides an easy way to set up and compare the quantization methods.
### Download models from hugging face
1. Prepare hugging face user access tokens
2. To get Llama-7b, use:
```shell
python download_models.py
```
3. To get Llama2-7b 13b and 70b, use:
```shell
python download_models_llama2.py
```

### Build the different quantization and run the benchmark codes! 
(Please check out the individual README files for each method. Alternatives are also provided below.)
#### ATOM 
1. Build and run the docker container
```shell
cd ./Atom
sudo docker build -t atom-image ./
sudo docker run --gpus all -v $(pwd)/Atom:/workspace/Atom \
 -v $(pwd)/llama_model:/workspace/llama_model \
 -v $(pwd)/llama_tokenizer:/workspace/llama_tokenizer \
 -v ~/.cache/huggingface:/root/.cache/huggingface \
 -it atom-image
```
#### QuaRot
1. Build and run the docker container
```shell
sudo docker build --no-cache -t quarot_image .
sudo docker run --rm -it \
  --gpus all \
  -v ./QuaRot:/workspace/QuaRot \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quarot_image
```
#### QServe


## Benchmark Results on RTX 6000
