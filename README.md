# Soft Self-Consistency Improves Language Model Agents
Code for the paper [Soft Self-Consistency Improves Language Model Agents](https://arxiv.org/abs/2402.13212) (ACL 2024).

## Requirements
You can install required packages by running the following command:
```bash
conda create -n softsc python=3.9
conda activate softsc
pip install -r requirements.txt
```
You may get issues when using `bitsandbytes`, my suggestion is to execute this command: `conda install nvidia::cudatoolkit==11.1.74`.

### Prepare the Environment for Bash
Please follow the setup in [InterCode](https://github.com/princeton-nlp/intercode) to install InterCode (recommended to build from source) and create the docker images for the Bash environment.

**Attention**: InterCode has four bash subsets and each subset has its own docker image. Please make sure to create all four docker images and each docker image should be named as `intercode-nl2bash{i}`, where `i` is the subset number. (Tips: you need to modify  `ENV file_system_version={i}` in `intercode/docker/nl2bash.Dockerfile` to create the corresponding docker image.)

### Prepare the Environment for Webshop
Please follow the setup instructions in [WebShop](https://github.com/princeton-nlp/WebShop) to install WebShop (recommended to use flag `-d all` to load all products) and launch the environment in a separate conda environment.

### Prepare the Environment for ALFWorld
Please follow the installation of [ALFWorld](https://github.com/alfworld/alfworld) **from source** to setup the environment. To use extra functionality used in this work, please replace `alfworld/agents/environment/alfred_tw_env.py` file with the one provided in [ADaPT](https://github.com/archiki/ADaPT).

## Run Experiments

### For Bash
You can run our code following the example below:
```bash
python run_bash.py \
    --model codellama/CodeLlama-7b-Instruct-hf \
    --cache_dir /path/to/cache_dir/of/model \
    --load_in_4bit \
    --k 20 \
    --aggregation mean \
    --method SoftSC \
    --output_dir /path/to/output_dir
```
We explain some import arguments below (some are not shown in the example):
- `k`: the number of samples.
- `aggregation`: the aggregation method for the samples. We support `mean`, `min`, and `prod`.
- `method`: the method to use. We support Self-Consistency `SC`, Soft Self-Consistency `SoftSC`, Adaptive Self-Consistency `AdaptiveSC`, and Adaptive Soft Self-Consistency `AdaptiveSoftSC`.
- `threshold` (only for Adaptive methods): the threshold for the Adaptive methods `AdaptiveSC` and `AdaptiveSoftSC`.

You also can score the your own bash output file by following the example below:
```bash
python score_eval_bash.py \
    --model codellama/CodeLlama-7b-Instruct-hf \
    --cache_dir /path/to/cache_dir/of/model \
    --input_file /path/to/your_own_bash_output_file \
    --load_in_4bit \
    --aggregation mean \
    --method SoftSC \
    --output_dir /path/to/output_dir
```
Your own bash output file should be in the format like:
```json
{
    "0": {
        "query": "Calculate a list of duplicate md5 sum hashes for all the \".java\" files in the /testbed directory",
        "actions": [
            "find /testbed -name \"*.java\" -exec md5sum {} \\; | awk '{print $1}' | sort | uniq -d",
            "find /testbed -type f -name \"*.java\" -exec md5sum {} \\; | cut -d' ' -f1",
            "find /testbed -type f -name \"*.java\" | xargs md5sum | awk '{print $1}' | sort | uniq -d",
            "find /testbed -type f -name \"*.java\" -exec md5sum {} + | sort | uniq -d -w 32",
        ]
    }
    "1": {
    ...
    }
    ...
}
```

### For Webshop
You can generate 20 trajectories for selecting and buying products for each task instance by running:
```bash
python run_webshop.py --k 20 --start 0 --num_sess 50
```

You also can score your own WebShop output file by following the example below:
```bash
python score_eval_webshop.py \
    --model codellama/CodeLlama-7b-Instruct-hf \
    --cache_dir /path/to/cache_dir/of/model \
    --data_path /path/to/your_own_webshop_output_file \
    --load_in_4bit \
    --k 20 \
    --aggregation mean \
    --method SoftSC \
    --output_dir /path/to/output_dir
```

### For ALFWorld
In ALFWorld, we sample multiple actions for each step and select one, therefore, scoring is performed implicitly. To run ALFWorld on the test set, use:
```bash
python run_alfworld.py --k 10 --eval-all True \
--score avg --fname test_results --LM llama
```
For running self-consistency set score as `majority`.
## Acknowledgement
We sincerely thank the authors of [InterCode](https://github.com/princeton-nlp/intercode), [WebShop](https://github.com/princeton-nlp/WebShop), and [ALFWorld](https://github.com/alfworld/alfworld) for their contributions!

## Citation
```bibtex
@inproceedings{wang-etal-2024-soft,
    title={Soft Self-Consistency Improves Language Model Agents}, 
    author={Han Wang and Archiki Prasad and Elias Stengel-Eskin and Mohit Bansal},
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    year={2024},
}
```
