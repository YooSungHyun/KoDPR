import logging
import math
import os
from logging import StreamHandler
from typing import Optional, Union

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import wandb
from arguments.training_args import TrainingArguments
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
from utils.data.custom_sampler import DistributedUniqueSampler
from torch.cuda.amp import autocast
from torch_optimizer import Adafactor
from transformers import DebertaV2Model, DebertaV2Config

# it is only lstm example.
torch.backends.cudnn.enabled = False

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


# TODO(User): override training_step and eval_loop for your style
class DSTrainer(Trainer):
    def __init__(
        self,
        device_id,
        eval_metric=None,
        precision="fp32",
        cmd_logger=None,
        web_logger=None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        chk_addr_dict: dict = None,
        non_blocking: bool = True,
        log_every_n: int = 1,
        max_norm: float = 0.0,
        metric_on_cpu: bool = False,
    ):
        super().__init__(
            device_id,
            eval_metric,
            precision,
            cmd_logger,
            web_logger,
            max_epochs,
            max_steps,
            grad_accum_steps,
            limit_train_batches,
            limit_val_batches,
            validation_frequency,
            checkpoint_dir,
            checkpoint_frequency,
            chk_addr_dict,
            non_blocking,
            log_every_n,
            max_norm,
            metric_on_cpu,
        )

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

    def training_step(self, model, batch, batch_idx) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: model to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        # TODO(User): fit the input and output for your model architecture!
        with autocast(enabled=self.mixed_precision, dtype=self.precision):
            batch_p, batch_q = batch
            p_emb = model(batch_p, "passage")  # bsz x bert_dim
            q_emb = model(batch_q, "query")  # bsz x bert_dim
            pred = torch.matmul(q_emb, p_emb.T)  # bsz x bsz
            loss = self.ibn_loss(pred)
            correct, bsz = self.batch_acc(pred)

        def on_before_backward(loss):
            pass

        on_before_backward(loss)
        model.backward(loss)

        def on_after_backward():
            pass

        on_after_backward()

        log_output = {"loss": loss, "correct": correct, "bsz": bsz}
        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(log_output, dtype=torch.Tensor, function=lambda x: x.detach())

        web_log_every_n(
            self.web_logger,
            {
                "train/loss": self._current_train_return["loss"],
                "train/acc": self._current_train_return["correct"] / self._current_train_return["bsz"],
                "train/step": self.step,
                "train/global_step": self.global_step,
                "train/epoch": self.current_epoch,
            },
            self.step,
            self.log_every_n,
            self.device_id,
        )
        return self._current_train_return

    def eval_loop(
        self,
        model,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: model
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        def on_start_eval(model):
            model.eval()
            # requires_grad = True, but loss.backward() raised error
            # because grad_fn is None
            torch.set_grad_enabled(False)

        on_start_eval(model)

        def on_validation_epoch_start():
            pass

        if self.device_id == 0:
            iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")
            pbar = enumerate(iterable)
        else:
            pbar = enumerate(val_loader)

        eval_step = 0
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
            # Therefore, all sections except EVAL EPOCH END only output the value of rank 0.
            tensor_dict_to_device(batch, self.device, non_blocking=self.non_blocking)
            # I use distributed dataloader and wandb log only rank:0, and epoch loss all gather

            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            def on_validation_batch_start(batch, batch_idx):
                pass

            on_validation_batch_start(batch, batch_idx)

            # TODO(User): fit the input and output for your model architecture!
            with autocast(enabled=self.mixed_precision, dtype=self.precision):
                # if self.precision == torch.bfloat16:
                # tensor_dict_to_dtype(batch, self.precision)
                batch_p, batch_q = batch
                p_emb = model(batch_p, "passage")  # bsz x bert_dim
                q_emb = model(batch_q, "query")  # bsz x bert_dim
                pred = torch.matmul(q_emb, p_emb.T)  # bsz x bsz
                loss = self.ibn_loss(pred)
                correct, bsz = self.batch_acc(pred)

            # contrastive loss는 배치의 index로 label을 정하기 때문에, 무작정 gather하면 label이 안맞는 현상이 발생할 수 있다.
            tot_batch_loss.append(loss.to(metric_on_device))
            tot_batch_size.append(bsz)
            tot_batch_corr.append(correct.to(metric_on_device))

            log_output = {"loss": loss, "correct": correct, "bsz": bsz}
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            self._current_val_return = apply_to_collection(
                log_output, torch.Tensor, lambda x: x.detach().to(metric_on_device)
            )

            def on_validation_batch_end(eval_out, batch, batch_idx):
                pass

            on_validation_batch_end(pred, batch, batch_idx)

            web_log_every_n(
                self.web_logger,
                {
                    "eval_step/loss": self._current_val_return["loss"],
                    "eval_step/acc": self._current_val_return["correct"] / self._current_val_return["bsz"],
                    "eval_step/step": eval_step,
                    "eval_step/global_step": self.global_step,
                    "eval_step/epoch": self.current_epoch,
                },
                eval_step,
                self.log_every_n,
                self.device_id,
            )
            if self.device_id == 0:
                self._format_iterable(iterable, self._current_val_return, "val")
            eval_step += 1

        # TODO(User): Create any form you want to output to wandb!
        def on_validation_epoch_end(tot_batch_loss, tot_batch_size, tot_batch_corr, metric_device):
            # if you want to see all_reduce example, see `fsdp_train.py`'s eval_loop
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
            epoch_loss = torch.mean(loss_gathered_data)
            # epoch monitoring is must doing every epoch
            web_log_every_n(
                self.web_logger,
                {
                    "eval/loss": epoch_loss,
                    "eval_step/acc": tot_batch_corr / tot_batch_size,
                    "eval/epoch": self.current_epoch,
                },
                self.current_epoch,
                1,
                self.device_id,
            )

        on_validation_epoch_end(tot_batch_loss, tot_batch_size, tot_batch_corr, metric_on_device)

        def on_validation_model_train(model):
            torch.set_grad_enabled(True)
            model.train()

        on_validation_model_train(model)


def main(hparams: TrainingArguments):
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

    web_logger = None
    if local_rank == 0:
        web_logger = wandb.init(config=hparams)

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

    train_dataset = JsonlDataset(hparams.train_datasets_path, transform=preprocess)
    eval_dataset = JsonlDataset(hparams.eval_datasets_path, transform=preprocess)

    custom_train_sampler = DistributedUniqueSampler(
        dataset=train_dataset,
        batch_size=hparams.per_device_train_batch_size,
        tokenizer=tokenizer,
        num_replicas=world_size,
        rank=local_rank,
        seed=hparams.seed,
    )
    custom_eval_sampler = DistributedSampler(
        dataset=eval_dataset, num_replicas=world_size, rank=local_rank, seed=hparams.seed, shuffle=False
    )
    hparams.per_device_train_batch_size = hparams.per_device_train_batch_size * 2
    test = list(custom_train_sampler.__iter__())
    for i in range(0, len(test), hparams.per_device_train_batch_size):
        assert len(set(train_dataset[test[i : i + hparams.per_device_train_batch_size]]["answer"])) == len(
            train_dataset[test[i : i + hparams.per_device_train_batch_size]]["answer"]
        ), "answer가 중복되는 배치 발생!"

    # DataLoader's shuffle: one device get random indices dataset in every epoch
    # example np_dataset is already set (feature)7:1(label), so, it can be all shuffle `True` between sampler and dataloader
    train_dataloader = CustomDataLoader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=hparams.per_device_train_batch_size,
        sampler=custom_train_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
        pin_memory=True,
        persistent_workers=True,
    )

    eval_dataloader = CustomDataLoader(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=hparams.per_device_eval_batch_size,
        sampler=custom_eval_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
        pin_memory=True,
        persistent_workers=True,
    )

    # dataloader already calculate len(total_data) / (batch_size * dist.get_world_size())
    # accumulation is always floor
    steps_per_epoch = math.floor(len(train_dataloader) / hparams.accumulate_grad_batches)

    p_encoder = DebertaV2Model.from_pretrained(hparams.transformers_model_name)
    q_encoder = DebertaV2Model.from_pretrained(hparams.transformers_model_name)

    # Instantiate objects
    model = KobertBiEncoder(passage_encoder=p_encoder, query_encoder=q_encoder).cuda(local_rank)

    if local_rank == 0:
        web_logger.watch(model, log_freq=hparams.log_every_n)

    optimizer = None
    initial_lr = hparams.learning_rate / hparams.div_factor

    # If you use torch optim and scheduler, It can have unexpected behavior. The current implementation is written for the worst case scenario.
    # For some reason, I was found that `loss` is not `auto_cast`, so in the current example, `auto_cast` manually.
    # and BF16 `auto_cast` is not supported now (https://github.com/microsoft/DeepSpeed/issues/4772) it is manually implement too.
    # The optimizer will use zero_optimizer as normal, and the grad_scaler is expected to behave normally, since the id check is done.
    # https://github.com/microsoft/DeepSpeed/issues/4908
    # optimizer = Adafactor(
    #     model.parameters(),
    #     lr=hparams.learning_rate,
    #     beta1=hparams.optim_beta1,
    #     weight_decay=hparams.weight_decay,
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )
    cycle_momentum = True
    if isinstance(optimizer, Adafactor):
        cycle_momentum = False
    # TODO(user): If you want to using deepspeed lr_scheduler, change this code line
    # max_lr = hparams.learning_rate
    # initial_lr = hparams.learning_rate / hparams.div_factor
    # min_lr = hparams.learning_rate / hparams.final_div_factor
    scheduler = None
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        pct_start=hparams.warmup_ratio,
        epochs=hparams.max_epochs,
        div_factor=hparams.div_factor,
        final_div_factor=hparams.final_div_factor,
        steps_per_epoch=steps_per_epoch,
        cycle_momentum=cycle_momentum,
    )

    # If you want to using own optimizer and scheduler,
    # check config `zero_allow_untested_optimizer` and `zero_force_ds_cpu_optimizer`
    ds_config = json_to_dict(hparams.deepspeed_config)

    update_auto_nested_dict(ds_config, "lr", initial_lr)
    update_auto_nested_dict(ds_config, "train_micro_batch_size_per_gpu", hparams.per_device_train_batch_size)
    update_auto_nested_dict(ds_config, "gradient_accumulation_steps", hparams.accumulate_grad_batches)
    if "fp16" in ds_config.keys() and ds_config["fp16"]["enabled"]:
        hparams.model_dtype = "fp16"
    elif "bf16" in ds_config.keys() and ds_config["bf16"]["enabled"]:
        hparams.model_dtype = "bf16"
    else:
        hparams.model_dtype = "fp32"

    # Since the deepspeed lr scheduler is, after all, just a generic object-inherited custom scheduler, Only authorize the use of torch scheduler.
    # Also, the ZeroOptimizer.param_groups address is the same as the torch scheduler.optimizer.param_groups address.
    # Therefore, there is absolutely no reason to use the lr_scheduler provided by Deepspeed.
    assert (
        scheduler is not None or "scheduler" not in ds_config.keys()
    ), "Don't use Deepspeed Scheduler!!!!, It is so confused. Plz implement something!"

    if optimizer is not None:
        from deepspeed.runtime.zero.utils import is_zero_supported_optimizer

        if not is_zero_supported_optimizer(optimizer):
            ds_config.update({"zero_allow_untested_optimizer": True})
        if "zero_optimization" in ds_config.keys():
            if "offload_optimizer" in ds_config["zero_optimization"].keys():
                # custom optimizer and using cpu offload
                ds_config.update({"zero_force_ds_cpu_optimizer": False})

    # 0: model, 1: optimizer, 2: dataloader, 3: lr scheduler가 나온다
    # dataloader는 deepspeed에서 권장하는 세팅이지만, 어짜피 distributedsampler 적용된 놈이 나온다.
    # if optimizer and scheduler is None, it is initialized by ds_config
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        dist_init_required=True,
        config=ds_config,
    )

    # monitor: ReduceLROnPlateau scheduler is stepped using loss, so monitor input train or val loss
    lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": None}
    assert id(scheduler) == id(lr_scheduler["scheduler"]), "scheduler mismatch! plz check!!!!"
    assert (
        id(optimizer.param_groups[0])
        == id(lr_scheduler["scheduler"].optimizer.param_groups[0])
        == id(model.optimizer.param_groups[0])
    ), "optimizer is something changed check id!"
    criterion = torch.nn.MSELoss()

    # I think some addr is same into trainer init&fit respectfully
    chk_addr_dict = {
        "train_dataloader": id(train_dataloader),
        "eval_dataloader": id(eval_dataloader),
        "model": id(model),
        "optimizer": id(optimizer.param_groups),
        "criterion": id(criterion),
        "scheduler_cfg": id(lr_scheduler),
        "scheduler_cfg[scheduler]": id(lr_scheduler["scheduler"]),
        "scheduler_cfg[scheduler].optimizer.param_groups": id(lr_scheduler["scheduler"].optimizer.param_groups),
    }

    log_str = f"""\n##########################################
    train_dataloader addr: {chk_addr_dict["train_dataloader"]}
    eval_dataloader addr: {chk_addr_dict["eval_dataloader"]}
    model addr: {chk_addr_dict["model"]}
    optimizer addr: {chk_addr_dict["optimizer"]}
    criterion addr: {chk_addr_dict["criterion"]}
    scheduler_cfg addr: {chk_addr_dict["scheduler_cfg"]}
    scheduler addr: {chk_addr_dict["scheduler_cfg[scheduler]"]}
    scheduler's optimizer value addr: {chk_addr_dict["scheduler_cfg[scheduler].optimizer.param_groups"]}
    ##########################################
    """
    logger.debug(log_str)
    # TODO(User): input your eval_metric
    eval_metric = None
    trainer = DSTrainer(
        device_id=local_rank,
        eval_metric=eval_metric,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        web_logger=web_logger,
        max_epochs=hparams.max_epochs,
        grad_accum_steps=hparams.accumulate_grad_batches,
        chk_addr_dict=chk_addr_dict,
        checkpoint_dir=hparams.output_dir,
        log_every_n=hparams.log_every_n,
        max_norm=hparams.max_norm,
        metric_on_cpu=hparams.metric_on_cpu,
    )
    tokenizer.save_pretrained(hparams.output_dir)
    DebertaV2Config.from_pretrained(hparams.transformers_model_name).save_pretrained(hparams.output_dir)
    trainer.eval_loop(model=model, val_loader=eval_dataloader)
    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler_cfg=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        ckpt_path=hparams.output_dir,
    )

    if local_rank == 0:
        web_logger.finish(exit_code=0)


if __name__ == "__main__":
    assert torch.distributed.is_available(), "DDP is only multi gpu!! check plz!"
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_args")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
