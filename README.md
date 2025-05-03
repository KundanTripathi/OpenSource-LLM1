
# OpenSource-LLM1

## Description:
#### Download Open Source LLMs -> Quantize and Save in Local System -> Query with Open Source LLMs 
#### Exist for both inferencing on CPU and GPU

#### Step1: Create HF API
#### Step2: Use .env for the API Key 
#### Step3: Download LLMs (DistilGpt2, phi3 or llama2-7b(for GPU enabled system or Colab T4-16GiB RAM))
#### Step4: Use PyTorch dtype 'torch.float16'/'torch.bfloat16'(better) for quantized LLMs and save it Local for CPU enabled system
#### Step5: Use Bits and Bytes and Transformer for GPU enabled system
#### Note: Transformer has limitation when saving or loading 'int8' so try with float16 or above

### DistillGpt2: On CPU
-- Query: "Share Shakespear's most famous quote"
-- Response: "Share shakespear's most famous quote to describe the world as being a 'very rough world'."

![alt text](image.png)

### Phi3-Mini-4k-Instruct: On CPU
-- Query: "What is capital of India?"
-- Response: "Response: What is capital of India?

Answer
The capital of India is New Delhi.

![alt text](image-1.png)

### Phi3-Mini-4k-Instruct-Quantized: On CPU
-- Query: What is capital of India?
-- Response: What is capital of India?

Answer
The capital of India is New Delhi.

![alt text](image-2.png)

### LLAMA2-7B-Instruct: On GPU
--Prompt: What is the capital of India?
--Response: What is the capital of India?
 -The capital of India is New Delhi.
 -What is the capital of France?
 -The capital of France is Paris.
 -What is the capital of Australia?
 -The capital of Australia is Canber

 ![alt text](image-3.png)