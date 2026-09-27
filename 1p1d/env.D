# f2 = DECODE pool (kv_consumer) — Qwen3.8-Flash-Next 1P1D PoC (plan §7 E1).
# Part of the 1P1D fork of MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark — AGPL-3.0-or-later.
# Identical to env.P except PORT/VLLM_HOST_IP/VLLM_NIXL_SIDE_CHANNEL_HOST/
# KV_TRANSFER_ROLE — matched flags across the seam (same image digest, same
# checkpoint snapshot, same template => the proven KV contract).
IMAGE="vllm/vllm-openai:qwen38-flash-next"
SERVED_MODEL_NAME="qwen3.8-flash-next"
ABLIT=0
YARN=1
MAX_MODEL_LEN=262144
MTP_NUM_SPECULATIVE_TOKENS=3
KV_TARGET_GIB=20
HOST_RESERVE_GIB=28
KV_CACHE_DTYPE=fp8
MAMBA_SSM_CACHE_DTYPE=bfloat16
MAX_NUM_SEQS=8
CUDAGRAPH_CAPTURE_SIZES=auto
MAX_NUM_BATCHED_TOKENS=2048
MTP_DRAFT_VOCAB=files/draft_vocab_en_code_47k.txt
MTP_K_SCHEDULE=
COMPILATION_MODE=0
MTP_DISABLE_BLOCK_DROP=1
CHAT_TEMPLATE=
HOST_SLACK_GIB=5
PORT=8100
KV_TRANSFER_ROLE=kv_consumer
EXTRA_DOCKER_ARGS="--device /dev/infiniband -e VLLM_USE_V2_MODEL_RUNNER=1 -e GLOO_SOCKET_IFNAME=enP2p1s0f1np1 -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 -e TP_SOCKET_IFNAME=enP2p1s0f1np1 -e VLLM_HOST_IP=192.168.100.2 -e UCX_TLS=rc,cuda_copy,sm,self -e UCX_NET_DEVICES=roceP2p1s0f1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=192.168.2.193 -e VLLM_SSM_CONV_STATE_LAYOUT=DS"
