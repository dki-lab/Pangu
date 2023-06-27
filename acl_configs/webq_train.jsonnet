local dataset = "webq";
local decoding_steps = 5;
local device = 6;
local training_option = 2;
local val_option = 2;
local eos = "[SEP]";
{
  "dataset_reader": {
    "type": "bottom_up",
    "dataset": dataset,
    "decoding_steps": decoding_steps,
    "training_option": training_option,
  },
  "validation_dataset_reader": {
    "type": "bottom_up",
    "dataset": dataset,
    "decoding_steps": decoding_steps,
//    "training_option": training_option,
    "training_option": val_option,
    "infer": true,
  },
  "train_data_path": "data/webqsp_0107.train.json",
  "model": {
    "type": "bottom_up",
    "training_option": training_option,
    "val_option": val_option,
    "dataset": dataset,
    "decoding_steps": decoding_steps,
    "loss_option": 1,
    "EOS": eos,
    "using_hf": true,
    "em_augmentation": true,
    "device": device,  // this is a new field. Be careful when using -r option for training
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "huggingface_transformer",
          "model_name": "bert-large-uncased",
          "pooling": true
       }
      }
    },
    "hidden_size": 1024,
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
    "num_epochs": 10,
    "validation_metric": "+EM",
    // "patience": 5,
    "cuda_device": device,
    "num_gradient_accumulation_steps": 8,
    "callbacks": [
      {
        "type": "track_epoch_callback"
      }
    ],
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
      "parameter_groups": [
        [["source_embedder"], {"lr": 2e-5}]
      ]
    }
//    "summary_interval": 1
  },
//    "distributed": {
//     "cuda_devices": [5, 6, 7]
//    }
}
