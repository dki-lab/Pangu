local dataset = "grail";
local decoding_steps = 5;
local device = 7;
local training_option = 2;
local val_option = 2;
local eos = "[SEP]";
{
  "dataset_reader": {
    "type": "bottom_up",
    "dataset": dataset,
    "decoding_steps": decoding_steps,
    "training_option": training_option
  },
  "validation_dataset_reader": {
    "type": "bottom_up",
    "dataset": dataset,
    "decoding_steps": decoding_steps,
    "training_option": val_option,
    "infer": true
  },
  "train_data_path": "data/grailqa_v1.0_train.json",
  "validation_data_path": "data/grailqa_v1.0_dev.json",
  "model": {
    "type": "bottom_up",
    "training_option": training_option,
    "val_option": val_option,
    "dataset": dataset,
    "decoding_steps": decoding_steps,
    "loss_option": 1,
    "EOS": eos,
//    "using_hf": true,
    "em_augmentation": true,
    "encoder_decoder": "t5-base",
    "device": device,
    "dropout": 0.5
  },
  "data_loader": {   // previously iterator
    "shuffle": true,
    "batch_size": 1
  },
  "validation_data_loader": {
    "shuffle": true,
    "batch_size": 1
  },
  "trainer": {
    "num_epochs": 30,
    "validation_metric": "+EM",
    "patience": 5,
    "cuda_device": device,
    "num_gradient_accumulation_steps": 8,
    "callbacks": [
      {
        "type": "track_epoch_callback"
      }
    ],
    "optimizer": {
      "type": "adam",
      "lr": 1e-4
    }
//    "summary_interval": 1
  },
    "distributed": {
     "cuda_devices": [4, 5, 6, 7]
    }
}
