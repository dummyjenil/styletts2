/usr/local/lib/python3.12/dist-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
StyleTTS2Model(
  (bert): CustomAlbert(
    (embeddings): AlbertEmbeddings(
      (word_embeddings): Embedding(178, 128, padding_idx=0)
      (position_embeddings): Embedding(512, 128)
      (token_type_embeddings): Embedding(2, 128)
      (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0, inplace=False)
    )
    (encoder): AlbertTransformer(
      (embedding_hidden_mapping_in): Linear(in_features=128, out_features=768, bias=True)
      (albert_layer_groups): ModuleList(
        (0): AlbertLayerGroup(
          (albert_layers): ModuleList(
            (0): AlbertLayer(
              (full_layer_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (attention): AlbertAttention(
                (attention_dropout): Dropout(p=0, inplace=False)
                (output_dropout): Dropout(p=0, inplace=False)
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              )
              (ffn): Linear(in_features=768, out_features=2048, bias=True)
              (ffn_output): Linear(in_features=2048, out_features=768, bias=True)
              (activation): NewGELUActivation()
              (dropout): Dropout(p=0, inplace=False)
            )
          )
        )
      )
    )
    (pooler): Linear(in_features=768, out_features=768, bias=True)
    (pooler_activation): Tanh()
  )
  (bert_encoder): Linear(in_features=768, out_features=128, bias=True)
  (text_encoder): TextEncoder(
    (actv): LeakyReLU(negative_slope=0.2)
    (embedding): Embedding(178, 128)
    (cnn): ModuleList(
      (0-1): 2 x Sequential(
        (0): Conv1d(128, 128, kernel_size=(5,), stride=(1,), padding=(2,))
        (1): LayerNorm()
        (2): LeakyReLU(negative_slope=0.2)
        (3): Dropout(p=0.2, inplace=False)
      )
    )
    (lstm): LSTM(128, 64, batch_first=True, bidirectional=True)
  )
  (predictor): ProsodyPredictor(
    (text_encoder): DurationEncoder(
      (lstms): ModuleList(
        (0): LSTM(256, 64, batch_first=True, bidirectional=True)
        (1): AdaLayerNorm(
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
        (2): LSTM(256, 64, batch_first=True, bidirectional=True)
        (3): AdaLayerNorm(
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
      )
    )
    (lstm): LSTM(256, 64, batch_first=True, bidirectional=True)
    (duration_proj): LinearNorm(
      (linear_layer): Linear(in_features=128, out_features=50, bias=True)
    )
    (shared): LSTM(256, 64, batch_first=True, bidirectional=True)
    (F0): ModuleList(
      (0): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (conv1): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
        (dropout): Dropout(p=0.2, inplace=False)
        (pool): Identity()
      )
      (1): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (conv1): Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=128, bias=True)
        )
        (conv1x1): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
        (dropout): Dropout(p=0.2, inplace=False)
        (pool): ConvTranspose1d(128, 128, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,), groups=128)
      )
      (2): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (conv1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=128, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=128, bias=True)
        )
        (dropout): Dropout(p=0.2, inplace=False)
        (pool): Identity()
      )
    )
    (N): ModuleList(
      (0): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (conv1): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
        (dropout): Dropout(p=0.2, inplace=False)
        (pool): Identity()
      )
      (1): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (conv1): Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=256, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=128, bias=True)
        )
        (conv1x1): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
        (dropout): Dropout(p=0.2, inplace=False)
        (pool): ConvTranspose1d(128, 128, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,), groups=128)
      )
      (2): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (conv1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=128, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=128, bias=True)
        )
        (dropout): Dropout(p=0.2, inplace=False)
        (pool): Identity()
      )
    )
    (F0_proj): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
    (N_proj): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
  )
  (style_encoder): StyleEncoder(
    (shared): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64)
        )
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv1x1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (2): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        )
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        )
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (4): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        )
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (5): LeakyReLU(negative_slope=0.2)
      (6): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1))
      (7): AdaptiveAvgPool2d(output_size=1)
      (8): LeakyReLU(negative_slope=0.2)
    )
    (unshared): Linear(in_features=128, out_features=128, bias=True)
  )
  (predictor_encoder): StyleEncoder(
    (shared): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64)
        )
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv1x1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (2): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        )
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        )
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (4): ResBlk(
        (actv): LeakyReLU(negative_slope=0.2)
        (downsample): DownSample()
        (downsample_res): LearnedDownSample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        )
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (5): LeakyReLU(negative_slope=0.2)
      (6): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1))
      (7): AdaptiveAvgPool2d(output_size=1)
      (8): LeakyReLU(negative_slope=0.2)
    )
    (unshared): Linear(in_features=128, out_features=128, bias=True)
  )
  (decoder): ISTFTDecoder(
    (encode): AdainResBlk1d(
      (actv): LeakyReLU(negative_slope=0.2)
      (dropout): Dropout(p=0.0, inplace=False)
      (conv1): Conv1d(130, 1024, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv2): Conv1d(1024, 1024, kernel_size=(3,), stride=(1,), padding=(1,))
      (norm1): AdaIN1d(
        (norm): InstanceNorm1d(130, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (fc): Linear(in_features=128, out_features=260, bias=True)
      )
      (norm2): AdaIN1d(
        (norm): InstanceNorm1d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (fc): Linear(in_features=128, out_features=2048, bias=True)
      )
      (conv1x1): Conv1d(130, 1024, kernel_size=(1,), stride=(1,), bias=False)
      (pool): Identity()
    )
    (decode): ModuleList(
      (0-2): 3 x AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv1): Conv1d(1090, 1024, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(1024, 1024, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(1090, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=2180, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=2048, bias=True)
        )
        (conv1x1): Conv1d(1090, 1024, kernel_size=(1,), stride=(1,), bias=False)
        (pool): Identity()
      )
      (3): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope=0.2)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv1): Conv1d(1090, 512, kernel_size=(3,), stride=(1,), padding=(1,))
        (conv2): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,))
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(1090, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=2180, bias=True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (fc): Linear(in_features=128, out_features=1024, bias=True)
        )
        (conv1x1): Conv1d(1090, 512, kernel_size=(1,), stride=(1,), bias=False)
        (pool): ConvTranspose1d(1090, 1090, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,), groups=1090)
      )
    )
    (F0_conv): Conv1d(1, 1, kernel_size=(3,), stride=(2,), padding=(1,))
    (N_conv): Conv1d(1, 1, kernel_size=(3,), stride=(2,), padding=(1,))
    (asr_res): Sequential(
      (0): Conv1d(512, 64, kernel_size=(1,), stride=(1,))
    )
    (generator): Generator(
      (m_source): SourceModuleHnNSF(
        (l_sin_gen): SineGen()
        (l_linear): Linear(in_features=9, out_features=1, bias=True)
        (l_tanh): Tanh()
      )
      (f0_upsamp): Upsample(scale_factor=300.0, mode='nearest')
      (ups): ModuleList(
        (0): ConvTranspose1d(128, 64, kernel_size=(20,), stride=(10,), padding=(5,))
        (1): ConvTranspose1d(64, 32, kernel_size=(12,), stride=(6,), padding=(3,))
      )
      (resblocks): ModuleList(
        (0): AdaINResBlock(
          (convs1): ModuleList(
            (0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
            (2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(5,), dilation=(5,))
          )
          (convs2): ModuleList(
            (0-2): 3 x Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
          )
          (adain1): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=128, bias=True)
            )
          )
          (adain2): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=128, bias=True)
            )
          )
          (alpha1): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x64x1]
              (1): Parameter containing: [torch.float32 of size 1x64x1]
              (2): Parameter containing: [torch.float32 of size 1x64x1]
          )
          (alpha2): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x64x1]
              (1): Parameter containing: [torch.float32 of size 1x64x1]
              (2): Parameter containing: [torch.float32 of size 1x64x1]
          )
        )
        (1): AdaINResBlock(
          (convs1): ModuleList(
            (0): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,))
            (1): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(9,), dilation=(3,))
            (2): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(15,), dilation=(5,))
          )
          (convs2): ModuleList(
            (0-2): 3 x Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,))
          )
          (adain1): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=128, bias=True)
            )
          )
          (adain2): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=128, bias=True)
            )
          )
          (alpha1): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x64x1]
              (1): Parameter containing: [torch.float32 of size 1x64x1]
              (2): Parameter containing: [torch.float32 of size 1x64x1]
          )
          (alpha2): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x64x1]
              (1): Parameter containing: [torch.float32 of size 1x64x1]
              (2): Parameter containing: [torch.float32 of size 1x64x1]
          )
        )
        (2): AdaINResBlock(
          (convs1): ModuleList(
            (0): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
            (2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(5,), dilation=(5,))
          )
          (convs2): ModuleList(
            (0-2): 3 x Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
          )
          (adain1): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=64, bias=True)
            )
          )
          (adain2): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=64, bias=True)
            )
          )
          (alpha1): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x32x1]
              (1): Parameter containing: [torch.float32 of size 1x32x1]
              (2): Parameter containing: [torch.float32 of size 1x32x1]
          )
          (alpha2): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x32x1]
              (1): Parameter containing: [torch.float32 of size 1x32x1]
              (2): Parameter containing: [torch.float32 of size 1x32x1]
          )
        )
        (3): AdaINResBlock(
          (convs1): ModuleList(
            (0): Conv1d(32, 32, kernel_size=(7,), stride=(1,), padding=(3,))
            (1): Conv1d(32, 32, kernel_size=(7,), stride=(1,), padding=(9,), dilation=(3,))
            (2): Conv1d(32, 32, kernel_size=(7,), stride=(1,), padding=(15,), dilation=(5,))
          )
          (convs2): ModuleList(
            (0-2): 3 x Conv1d(32, 32, kernel_size=(7,), stride=(1,), padding=(3,))
          )
          (adain1): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=64, bias=True)
            )
          )
          (adain2): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=64, bias=True)
            )
          )
          (alpha1): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x32x1]
              (1): Parameter containing: [torch.float32 of size 1x32x1]
              (2): Parameter containing: [torch.float32 of size 1x32x1]
          )
          (alpha2): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x32x1]
              (1): Parameter containing: [torch.float32 of size 1x32x1]
              (2): Parameter containing: [torch.float32 of size 1x32x1]
          )
        )
      )
      (noise_convs): ModuleList(
        (0): Conv1d(22, 64, kernel_size=(np.int64(12),), stride=(np.int64(6),), padding=(np.int64(3),))
        (1): Conv1d(22, 32, kernel_size=(1,), stride=(1,))
      )
      (noise_res): ModuleList(
        (0): AdaINResBlock(
          (convs1): ModuleList(
            (0): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,))
            (1): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(9,), dilation=(3,))
            (2): Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(15,), dilation=(5,))
          )
          (convs2): ModuleList(
            (0-2): 3 x Conv1d(64, 64, kernel_size=(7,), stride=(1,), padding=(3,))
          )
          (adain1): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=128, bias=True)
            )
          )
          (adain2): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=128, bias=True)
            )
          )
          (alpha1): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x64x1]
              (1): Parameter containing: [torch.float32 of size 1x64x1]
              (2): Parameter containing: [torch.float32 of size 1x64x1]
          )
          (alpha2): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x64x1]
              (1): Parameter containing: [torch.float32 of size 1x64x1]
              (2): Parameter containing: [torch.float32 of size 1x64x1]
          )
        )
        (1): AdaINResBlock(
          (convs1): ModuleList(
            (0): Conv1d(32, 32, kernel_size=(11,), stride=(1,), padding=(5,))
            (1): Conv1d(32, 32, kernel_size=(11,), stride=(1,), padding=(15,), dilation=(3,))
            (2): Conv1d(32, 32, kernel_size=(11,), stride=(1,), padding=(25,), dilation=(5,))
          )
          (convs2): ModuleList(
            (0-2): 3 x Conv1d(32, 32, kernel_size=(11,), stride=(1,), padding=(5,))
          )
          (adain1): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=64, bias=True)
            )
          )
          (adain2): ModuleList(
            (0-2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (fc): Linear(in_features=128, out_features=64, bias=True)
            )
          )
          (alpha1): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x32x1]
              (1): Parameter containing: [torch.float32 of size 1x32x1]
              (2): Parameter containing: [torch.float32 of size 1x32x1]
          )
          (alpha2): ParameterList(
              (0): Parameter containing: [torch.float32 of size 1x32x1]
              (1): Parameter containing: [torch.float32 of size 1x32x1]
              (2): Parameter containing: [torch.float32 of size 1x32x1]
          )
        )
      )
      (conv_post): Conv1d(32, 22, kernel_size=(7,), stride=(1,), padding=(3,))
      (reflection_pad): ReflectionPad1d((1, 0))
      (stft): CustomSTFT()
    )
  )
  (diffusion): AudioDiffusionConditional(
    (unet): StyleTransformer1d(
      (blocks): ModuleList(
        (0-2): 3 x StyleTransformerBlock(
          (attn): StyleAttention(
            (norm): AdaLayerNorm(
              (fc): Linear(in_features=256, out_features=2048, bias=True)
            )
            (to_qkv): Linear(in_features=1024, out_features=1536, bias=False)
            (to_out): Linear(in_features=512, out_features=1024, bias=True)
          )
          (ff): Sequential(
            (0): Linear(in_features=1024, out_features=2048, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=2048, out_features=1024, bias=True)
          )
          (norm): AdaLayerNorm(
            (fc): Linear(in_features=256, out_features=2048, bias=True)
          )
        )
      )
      (to_out): Conv1d(1024, 256, kernel_size=(1,), stride=(1,))
      (to_mapping): Sequential(
        (0): Linear(in_features=1024, out_features=1024, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=1024, out_features=1024, bias=True)
        (3): GELU(approximate='none')
      )
      (to_time): Sequential(
        (0): Linear(in_features=1, out_features=1024, bias=True)
        (1): GELU(approximate='none')
      )
      (to_features): Sequential(
        (0): Linear(in_features=256, out_features=1024, bias=True)
        (1): GELU(approximate='none')
      )
    )
  )
  (text_aligner): ASRCNN(
    (to_mfcc): MFCC()
    (init_cnn): Conv1d(40, 256, kernel_size=(7,), stride=(2,), padding=(3,))
    (layers): ModuleList(
      (0-5): 6 x Sequential(
        (0): ConvBlock(
          (blocks): ModuleList(
            (0): Sequential(
              (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
              (1): ReLU()
              (2): GroupNorm(8, 256, eps=1e-05, affine=True)
              (3): Dropout(p=0.2, inplace=False)
              (4): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
              (5): ReLU()
              (6): Dropout(p=0.2, inplace=False)
            )
            (1): Sequential(
              (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
              (1): ReLU()
              (2): GroupNorm(8, 256, eps=1e-05, affine=True)
              (3): Dropout(p=0.2, inplace=False)
              (4): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
              (5): ReLU()
              (6): Dropout(p=0.2, inplace=False)
            )
            (2): Sequential(
              (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
              (1): ReLU()
              (2): GroupNorm(8, 256, eps=1e-05, affine=True)
              (3): Dropout(p=0.2, inplace=False)
              (4): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
              (5): ReLU()
              (6): Dropout(p=0.2, inplace=False)
            )
          )
        )
        (1): GroupNorm(1, 256, eps=1e-05, affine=True)
      )
    )
    (proj): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
    (ctc): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=178, bias=True)
    )
    (s2s): ASRS2S(
      (embed): Embedding(178, 256)
      (attention): Attention(
        (q_layer): Linear(in_features=128, out_features=128, bias=False)
        (m_layer): Linear(in_features=128, out_features=128, bias=False)
        (v): Linear(in_features=128, out_features=1, bias=False)
        (loc_conv): Conv1d(2, 32, kernel_size=(63,), stride=(1,), padding=(31,), bias=False)
        (loc_dense): Linear(in_features=32, out_features=128, bias=False)
      )
      (decoder_rnn): LSTMCell(384, 128)
      (proj_hidden): Sequential(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): Tanh()
      )
      (proj_tokens): Linear(in_features=128, out_features=178, bias=True)
    )
  )
  (pitch_extractor): JDCNet(
    (conv_block): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.01, inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
    (res_block1): ResBlock(
      (pre_conv): Sequential(
        (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): LeakyReLU(negative_slope=0.01, inplace=True)
        (2): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
      )
      (conv): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01, inplace=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (conv1by1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (res_block2): ResBlock(
      (pre_conv): Sequential(
        (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): LeakyReLU(negative_slope=0.01, inplace=True)
        (2): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
      )
      (conv): Sequential(
        (0): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01, inplace=True)
        (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (conv1by1): Conv2d(128, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (res_block3): ResBlock(
      (pre_conv): Sequential(
        (0): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): LeakyReLU(negative_slope=0.01, inplace=True)
        (2): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
      )
      (conv): Sequential(
        (0): Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01, inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (conv1by1): Conv2d(192, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (pool_block): Sequential(
      (0): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): LeakyReLU(negative_slope=0.01, inplace=True)
      (2): MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0, dilation=1, ceil_mode=False)
      (3): Dropout(p=0.2, inplace=False)
    )
    (bilstm_classifier): LSTM(512, 256, batch_first=True, bidirectional=True)
    (classifier): Linear(in_features=512, out_features=1, bias=True)
  )
  (mpd): MultiPeriodDiscriminator(
    (discriminators): ModuleList(
      (0-4): 5 x DiscriminatorP(
        (convs): ModuleList(
          (0): Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
          (1): Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
          (2): Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
          (3): Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
          (4): Conv2d(1024, 1024, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        )
        (conv_post): Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
      )
    )
  )
  (msd): MultiResSpecDiscriminator(
    (discriminators): ModuleList(
      (0-2): 3 x SpecDiscriminator(
        (discriminators): ModuleList(
          (0): Conv2d(1, 32, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4))
          (1-3): 3 x Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
          (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (out): Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
  )
  (wd): WavLMDiscriminator(
    (pre): Conv1d(9984, 64, kernel_size=(1,), stride=(1,))
    (convs): ModuleList(
      (0): Conv1d(64, 128, kernel_size=(5,), stride=(1,), padding=(2,))
      (1): Conv1d(128, 256, kernel_size=(5,), stride=(1,), padding=(2,))
      (2): Conv1d(256, 256, kernel_size=(5,), stride=(1,), padding=(2,))
    )
    (conv_post): Conv1d(256, 1, kernel_size=(3,), stride=(1,), padding=(1,))
  )
)