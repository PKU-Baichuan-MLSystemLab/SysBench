# export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=4,5

model="THUDM/glm-4-9b-chat"
vllm serve $model \
    --dtype auto \
    --port 33618 \
    --tensor-parallel-size 2 \
    --api-key custom-key \
    --trust-remote-code
