from dbs_data_loader import prepare_data_loader
from accelerate.state import AcceleratorState, DistributedType, GradientState, is_tpu_available

from accelerate import Accelerator

class AcceleratorWithDynamicBatchSampling(Accelerator):

    def __init__(self, dynamic_batch_sampler_args, *args, **kwargs):
        self.dynamic_batch_sampler_args=dynamic_batch_sampler_args
        super().__init__(*args, **kwargs)

    def prepare_data_loader(self, data_loader):

        return prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=self.device_placement if self.distributed_type != DistributedType.TPU else False,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
            dynamic_batch_sampler_args=self.dynamic_batch_sampler_args
        )
