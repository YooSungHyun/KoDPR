import logging
import math
import os
from logging import StreamHandler
from typing import Optional, Union

import deepspeed
import torch
import torch.distributed as dist
import wandb
from arguments.inference_args import InferenceArguments
from networks.models import KobertBiEncoder
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from trainer.deepspeed import Trainer
from utils.comfy import (
    dataclass_to_namespace,
    json_to_dict,
    seed_everything,
    update_auto_nested_dict,
    apply_to_collection,
    web_log_every_n,
    tensor_dict_to_device,
)
from transformers import AutoTokenizer
from utils.data.jsonl_dataset import JsonlDataset
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import DistributedUniqueBM25Sampler, DistributedUniqueSampler
from torch.cuda.amp import autocast
from transformers import DebertaV2Model, DebertaV2Config
from collections import Counter

from utils.model_checkpointing.ds_handler import load_checkpoint_for_infer

# it is only lstm example.
torch.backends.cudnn.enabled = False

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


class DSTrainer(Trainer):
    def __init__(
        self,
        device_id,
        precision="fp32",
        cmd_logger=None,
        chk_addr_dict: dict = None,
        non_blocking: bool = True,
        metric_on_cpu: bool = False,
    ):
        super().__init__(device_id, precision, cmd_logger, chk_addr_dict, non_blocking, metric_on_cpu)

    def ibn_loss(self, pred: torch.FloatTensor):
        """in-batch negative를 활용한 batch의 loss를 계산합니다.
        pred : bsz x bsz 또는 bsz x bsz*2의 logit 값을 가짐. 후자는 hard negative를 포함하는 경우.
        """
        bsz = pred.size(0)
        target = torch.arange(bsz).to(self.device)  # 주대각선이 answer
        return torch.nn.functional.cross_entropy(pred, target)

    def batch_acc(self, pred: torch.FloatTensor):
        """batch 내의 accuracy를 계산합니다."""
        bsz = pred.size(0)
        target = torch.arange(bsz)  # 주대각선이 answer
        return (pred.detach().cpu().max(1).indices == target).sum().float(), bsz

    def test_loop(self, model, test_loader: Optional[torch.utils.data.DataLoader], **kwargs):
        """The test loop ruunning a single test epoch.

        Args:
            model: model
            test_loader: The dataloader yielding the test batches.

        """
        # no test if test_loader wasn't passed
        if test_loader is None:
            return

        def on_start_test(model):
            model.test()
            # requires_grad = True, but loss.backward() raised error
            # because grad_fn is None
            torch.set_grad_enabled(False)

        on_start_test(model)

        def on_test_epoch_start():
            pass

        if self.device_id == 0:
            iterable = self.progbar_wrapper(test_loader, total=len(test_loader), desc="test")
            pbar = enumerate(iterable)
        else:
            pbar = enumerate(test_loader)

        test_step = 0
        tot_batch_loss = list()
        tot_batch_size = list()
        tot_batch_corr = list()

        if self.metric_on_cpu:
            metric_on_device = torch.device("cpu")
        else:
            metric_on_device = self.device

        for batch_idx, batch in pbar:
            # I tried to output the most accurate LOSS to WANDB with ALL_GATHER for all LOSS sections,
            # but it was not much different from outputting the value of GPU 0.
            # Therefore, all sections except test EPOCH END only output the value of rank 0.
            tensor_dict_to_device(batch, self.device, non_blocking=self.non_blocking)
            # I use distributed dataloader and wandb log only rank:0, and epoch loss all gather

            def on_test_batch_start(batch, batch_idx):
                pass

            on_test_batch_start(batch, batch_idx)

            with autocast(enabled=self.mixed_precision, dtype=self.precision):
                # if self.precision == torch.bfloat16:
                # tensor_dict_to_dtype(batch, self.precision)
                batch_p, batch_q = batch
                p_emb = model(batch_p, "passage")  # bsz x bert_dim
                q_emb = model(batch_q, "query")  # bsz x bert_dim
                pred = torch.matmul(q_emb, p_emb.T)  # bsz x bsz
                loss = self.ibn_loss(pred)
                correct, bsz = self.batch_acc(pred)

            tot_batch_loss.append(loss.to(metric_on_device))
            tot_batch_size.append(bsz)
            tot_batch_corr.append(correct.to(metric_on_device))

            log_output = {"loss": loss}
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            self._current_val_return = apply_to_collection(
                log_output, torch.Tensor, lambda x: x.detach().to(metric_on_device)
            )

            def on_test_batch_end(test_out, batch, batch_idx):
                pass

            on_test_batch_end(log_output, batch, batch_idx)

            if self.device_id == 0:
                self._format_iterable(iterable, self._current_val_return, "test")
            test_step += 1

        # TODO(User): Create any form you want to output to wandb!
        def on_test_epoch_end(tot_batch_loss, tot_batch_size, tot_batch_corr, metric_device):
            # if you want to see all_reduce example, see `fsdp_train.py`'s test_loop
            # if you want to see all_reduce example, see `fsdp_train.py`'s test_loop
            tot_batch_loss = torch.stack(tot_batch_loss)
            tot_batch_size = torch.tensor(tot_batch_size, device=metric_device)
            dist.all_reduce(tot_batch_size, dist.ReduceOp.SUM)
            tot_batch_size = torch.sum(tot_batch_size)
            tot_batch_corr = torch.stack(tot_batch_corr)
            dist.all_reduce(tot_batch_corr, dist.ReduceOp.SUM)
            tot_batch_corr = torch.sum(tot_batch_corr)
            # all_gather` requires a `fixed length tensor` as input.
            # Since the length of the data on each GPU may be different, the length should be passed to `all_gather` first.
            local_size = torch.tensor([tot_batch_loss.size(0)], dtype=torch.long, device=metric_device)
            size_list = [
                torch.tensor([0], dtype=torch.long, device=metric_device) for _ in range(dist.get_world_size())
            ]
            if metric_device == torch.device("cpu"):
                dist.all_gather_object(size_list, local_size)
            else:
                dist.all_gather(size_list, local_size)

            # Create a fixed length tensor with the length of `all_gather`.
            loss_gathered_data = [
                torch.zeros(size.item(), dtype=tot_batch_loss.dtype, device=metric_device) for size in size_list
            ]

            # Collect and match data from all GPUs.
            if metric_device == torch.device("cpu"):
                # Collect and match data from all GPUs.
                dist.all_gather_object(loss_gathered_data, tot_batch_loss)
            else:
                dist.all_gather(loss_gathered_data, tot_batch_loss)

            # example 4 gpus : [gpu0[tensor],gpu1[tensor],gpu2[tensor],gpu3[tensor]]
            loss_gathered_data = torch.cat(loss_gathered_data, dim=0)
            # epoch_loss = torch.mean(loss_gathered_data)
            # pd_result = pd.DataFrame(np_outputs, columns=["pred", "labels"])
            # pd_result.to_excel("./ds_result.xlsx", index=False)

        on_test_epoch_end(tot_batch_loss, tot_batch_size, tot_batch_corr, metric_on_device)


def main(hparams: InferenceArguments):
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    seed_everything(hparams.seed)
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    logger.info(
        f"Start running basic deepspeed example on total {world_size} computers, {rank}'s process on {local_rank}'s gpu."
    )

    assert world_size > -1 and rank > -1 and local_rank > -1, "Your distributed environ is wrong, plz check and retry!"

    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed("nccl", rank=rank, world_size=world_size)

    tokenizer = AutoTokenizer.from_pretrained(hparams.transformers_model_name)

    def preprocess(examples):
        # deberta는 cls, sep(eos) 자동으로 넣어줌
        batch_p = tokenizer(
            tokenizer.cls_token + examples["context"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        batch_q = tokenizer(
            tokenizer.cls_token + examples["question"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        examples["batch_p_input_ids"] = batch_p["input_ids"]
        examples["batch_q_input_ids"] = batch_q["input_ids"]

        return examples

    test_dataset = JsonlDataset(hparams.data_path, transform=preprocess)

    custom_test_sampler = DistributedUniqueSampler(
        dataset=test_dataset,
        batch_size=hparams.per_device_test_batch_size,
        num_replicas=world_size,
        rank=local_rank,
        seed=hparams.seed,
        shuffle=False,
    )

    test_chk = list(custom_test_sampler.__iter__())
    for i in range(0, len(test_chk), hparams.per_device_test_batch_size):
        assert (
            Counter(test_dataset[test_chk[i : i + hparams.per_device_test_batch_size]]["answer"]).most_common(1)[0][1]
            == 1
        ), "answer가 중복되는 배치 발생!"

    # DataLoader's shuffle: one device get random indices dataset in every epoch
    # example np_dataset is already set (feature)7:1(label), so, it can be all shuffle `True` between sampler and dataloader
    test_dataloader = CustomDataLoader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=hparams.per_device_test_batch_size,
        sampler=custom_test_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
        pin_memory=True,
        persistent_workers=True,
    )

    p_encoder = DebertaV2Model.from_pretrained(hparams.transformers_model_name)
    q_encoder = DebertaV2Model.from_pretrained(hparams.transformers_model_name)

    # Instantiate objects
    model = KobertBiEncoder(passage_encoder=p_encoder, query_encoder=q_encoder).cuda(local_rank)

    state = {"model": model}
    load_checkpoint_for_infer(
        state,
        checkpoint_filepath=hparams.model_path,
        model_file_name="mp_rank_00_model_states.pt",
        device=f"cuda:{local_rank}",
        logger=logger,
    )
    # Since the deepspeed lr scheduler is, after all, just a generic object-inherited custom scheduler, Only authorize the use of torch scheduler.
    # Also, the ZeroOptimizer.param_groups address is the same as the torch scheduler.optimizer.param_groups address.
    # Therefore, there is absolutely no reason to use the lr_scheduler provided by Deepspeed.

    # in deepspeed, precision is just using model log for .pt file
    precision_dict = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    precision = torch.float32
    if precision in ["fp16", "float16"]:
        precision = precision_dict["fp16"]
    elif precision in ["bf16" or "bfloat16"]:
        precision = precision_dict["bf16"]

    model = deepspeed.init_inference(
        model=state["model"],
        mp_size=world_size,
        dtype=precision,
        injection_policy={KobertBiEncoder: ("passage_encoder.encoder", "query_encoder.encoder")},
        replace_with_kernel_inject=False,
    )

    # monitor: ReduceLROnPlateau scheduler is stepped using loss, so monitor input train or val loss
    test_metric = None

    # TODO(User): input your test_metric
    test_metric = None
    trainer = DSTrainer(
        device_id=local_rank,
        test_metric=test_metric,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        metric_on_cpu=hparams.metric_on_cpu,
    )

    trainer.test_loop(model=model, test_loader=test_dataloader)


if __name__ == "__main__":
    assert torch.distributed.is_available(), "DDP is only multi gpu!! check plz!"
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="training_args")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
