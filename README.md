
# OpenSource-LLM1

## Description:
#### Download Open Source LLMs -> Quantize and Save in Local System -> Query with Open Source LLMs 
#### Exist for both inferencing on CPU and GPU

### Step1: Create HF API
### Step2: Use .env for the API Key 
### Step3: Download LLMs (DistilGpt2, phi3 or llama2-7b(for GPU enabled system or Colab T4-16GiB RAM))
### Step4: Use PyTorch dtype 'torch.float16'/'torch.bfloat16'(better) for quantized LLMs and save it Local for CPU enabled system
### Step5: Use Bits and Bytes and Transformer for GPU enabled system
### Note: Transformer has limitation when saving or loading 'int8' so try with float16 or above