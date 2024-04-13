# mixtral.c

Inference of Mixtral models in pure C

Inspired by and using code from [llama2.c](https://github.com/karpathy/llama2.c)

## Start

```
python3 tokenizer.py -t <model_path>
python3 export.py mixtral.bin --model <model_path>
make runfast
./run mixtral.bin
```
