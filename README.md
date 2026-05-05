```javascript
StyleTTS2Model(
  (bert): CustomAlbert(
    (embeddings): AlbertEmbeddings(
      (word_embeddings): Embedding(178, 128, padding_idx = 0)
      (position_embeddings): Embedding(512, 128)
      (token_type_embeddings): Embedding(2, 128)
      (LayerNorm): LayerNorm((128,), eps = 1e-12, elementwise_affine = True)
      (dropout): Dropout(p = 0, inplace = False)
    )
    (encoder): AlbertTransformer(
      (embedding_hidden_mapping_in): Linear(in_features = 128, out_features = 768, bias = True)
      (albert_layer_groups): ModuleList(
        (0): AlbertLayerGroup(
          (albert_layers): ModuleList(
            (0): AlbertLayer(
              (full_layer_layer_norm): LayerNorm((768,), eps = 1e-12, elementwise_affine = True)
              (attention): AlbertAttention(
                (attention_dropout): Dropout(p = 0, inplace = False)
                (output_dropout): Dropout(p = 0, inplace = False)
                (query): Linear(in_features = 768, out_features = 768, bias = True)
                (key): Linear(in_features = 768, out_features = 768, bias = True)
                (value): Linear(in_features = 768, out_features = 768, bias = True)
                (dense): Linear(in_features = 768, out_features = 768, bias = True)
                (LayerNorm): LayerNorm((768,), eps = 1e-12, elementwise_affine = True)
              )
              (ffn): Linear(in_features = 768, out_features = 2048, bias = True)
              (ffn_output): Linear(in_features = 2048, out_features = 768, bias = True)
              (activation): NewGELUActivation()
              (dropout): Dropout(p = 0, inplace = False)
            )
          )
        )
      )
    )
    (pooler): Linear(in_features = 768, out_features = 768, bias = True)
    (pooler_activation): Tanh()
  )
  (bert_encoder): Linear(in_features = 768, out_features = 128, bias = True)
  (text_encoder): TextEncoder(
    (actv): LeakyReLU(negative_slope = 0.2)
    (embedding): Embedding(178, 128)
    (cnn): ModuleList(
      (0 - 1): 2 x Sequential(
        (0): ParametrizedConv1d(
          128, 128, kernel_size = (5,), stride = (1,), padding = (2,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (1): LayerNorm()
        (2): LeakyReLU(negative_slope = 0.2)
        (3): Dropout(p = 0.2, inplace = False)
      )
    )
    (lstm): LSTM(128, 64, batch_first = True, bidirectional = True)
  )
  (predictor): ProsodyPredictor(
    (text_encoder): DurationEncoder(
      (lstms): ModuleList(
        (0): LSTM(256, 64, batch_first = True, bidirectional = True)
        (1): AdaLayerNorm(
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
        (2): LSTM(256, 64, batch_first = True, bidirectional = True)
        (3): AdaLayerNorm(
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
      )
    )
    (lstm): LSTM(256, 64, batch_first = True, bidirectional = True)
    (duration_proj): LinearNorm(
      (linear_layer): Linear(in_features = 128, out_features = 50, bias = True)
    )
    (shared): LSTM(256, 64, batch_first = True, bidirectional = True)
    (F0): ModuleList(
      (0): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (conv1): ParametrizedConv1d(
          128, 128, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          128, 128, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(128, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
        (dropout): Dropout(p = 0.2, inplace = False)
        (pool): Identity()
      )
      (1): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (conv1): ParametrizedConv1d(
          128, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 128, bias = True)
        )
        (conv1x1): ParametrizedConv1d(
          128, 64, kernel_size = (1,), stride = (1,), bias = False
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (dropout): Dropout(p = 0.2, inplace = False)
        (pool): ParametrizedConvTranspose1d(
          128, 128, kernel_size = (3,), stride = (2,), padding = (1,), output_padding = (1,), groups = 128
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
      )
      (2): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (conv1): ParametrizedConv1d(
          64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 128, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 128, bias = True)
        )
        (dropout): Dropout(p = 0.2, inplace = False)
        (pool): Identity()
      )
    )
    (N): ModuleList(
      (0): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (conv1): ParametrizedConv1d(
          128, 128, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          128, 128, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(128, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
        (dropout): Dropout(p = 0.2, inplace = False)
        (pool): Identity()
      )
      (1): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (conv1): ParametrizedConv1d(
          128, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(128, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 256, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 128, bias = True)
        )
        (conv1x1): ParametrizedConv1d(
          128, 64, kernel_size = (1,), stride = (1,), bias = False
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (dropout): Dropout(p = 0.2, inplace = False)
        (pool): ParametrizedConvTranspose1d(
          128, 128, kernel_size = (3,), stride = (2,), padding = (1,), output_padding = (1,), groups = 128
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
      )
      (2): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (conv1): ParametrizedConv1d(
          64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 128, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 128, bias = True)
        )
        (dropout): Dropout(p = 0.2, inplace = False)
        (pool): Identity()
      )
    )
    (F0_proj): Conv1d(64, 1, kernel_size = (1,), stride = (1,))
    (N_proj): Conv1d(64, 1, kernel_size = (1,), stride = (1,))
  )
  (decoder): ISTFTDecoder(
    (encode): AdainResBlk1d(
      (actv): LeakyReLU(negative_slope = 0.2)
      (dropout): Dropout(p = 0.0, inplace = False)
      (conv1): ParametrizedConv1d(
        130, 1024, kernel_size = (3,), stride = (1,), padding = (1,)
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
      )
      (conv2): ParametrizedConv1d(
        1024, 1024, kernel_size = (3,), stride = (1,), padding = (1,)
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
      )
      (norm1): AdaIN1d(
        (norm): InstanceNorm1d(130, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
        (fc): Linear(in_features = 128, out_features = 260, bias = True)
      )
      (norm2): AdaIN1d(
        (norm): InstanceNorm1d(1024, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
        (fc): Linear(in_features = 128, out_features = 2048, bias = True)
      )
      (conv1x1): ParametrizedConv1d(
        130, 1024, kernel_size = (1,), stride = (1,), bias = False
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
      )
      (pool): Identity()
    )
    (decode): ModuleList(
      (0 - 2): 3 x AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (dropout): Dropout(p = 0.0, inplace = False)
        (conv1): ParametrizedConv1d(
          1090, 1024, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          1024, 1024, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(1090, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 2180, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(1024, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 2048, bias = True)
        )
        (conv1x1): ParametrizedConv1d(
          1090, 1024, kernel_size = (1,), stride = (1,), bias = False
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (pool): Identity()
      )
      (3): AdainResBlk1d(
        (actv): LeakyReLU(negative_slope = 0.2)
        (dropout): Dropout(p = 0.0, inplace = False)
        (conv1): ParametrizedConv1d(
          1090, 512, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (conv2): ParametrizedConv1d(
          512, 512, kernel_size = (3,), stride = (1,), padding = (1,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (norm1): AdaIN1d(
          (norm): InstanceNorm1d(1090, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 2180, bias = True)
        )
        (norm2): AdaIN1d(
          (norm): InstanceNorm1d(512, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
          (fc): Linear(in_features = 128, out_features = 1024, bias = True)
        )
        (conv1x1): ParametrizedConv1d(
          1090, 512, kernel_size = (1,), stride = (1,), bias = False
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (pool): ParametrizedConvTranspose1d(
          1090, 1090, kernel_size = (3,), stride = (2,), padding = (1,), output_padding = (1,), groups = 1090
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
      )
    )
    (F0_conv): ParametrizedConv1d(
      1, 1, kernel_size = (3,), stride = (2,), padding = (1,)
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _WeightNorm()
          )
        )
    )
    (N_conv): ParametrizedConv1d(
      1, 1, kernel_size = (3,), stride = (2,), padding = (1,)
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _WeightNorm()
          )
        )
    )
    (asr_res): Sequential(
      (0): ParametrizedConv1d(
        512, 64, kernel_size = (1,), stride = (1,)
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
      )
    )
    (generator): Generator(
      (m_source): SourceModuleHnNSF(
        (l_sin_gen): SineGen()
        (l_linear): Linear(in_features = 9, out_features = 1, bias = True)
        (l_tanh): Tanh()
      )
      (f0_upsamp): Upsample(scale_factor = 300.0, mode = 'nearest')
      (ups): ModuleList(
        (0): ParametrizedConvTranspose1d(
          128, 64, kernel_size = (20,), stride = (10,), padding = (5,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
        (1): ParametrizedConvTranspose1d(
          64, 32, kernel_size = (12,), stride = (6,), padding = (3,)
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
        )
      )
      (resblocks): ModuleList(
        (0): AdaINResBlock(
          (convs1): ModuleList(
            (0): ParametrizedConv1d(
              64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (1): ParametrizedConv1d(
              64, 64, kernel_size = (3,), stride = (1,), padding = (3,), dilation = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (2): ParametrizedConv1d(
              64, 64, kernel_size = (3,), stride = (1,), padding = (5,), dilation = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (convs2): ModuleList(
            (0 - 2): 3 x ParametrizedConv1d(
              64, 64, kernel_size = (3,), stride = (1,), padding = (1,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (adain1): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 128, bias = True)
            )
          )
          (adain2): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 128, bias = True)
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
            (0): ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (1): ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (9,), dilation = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (2): ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (15,), dilation = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (convs2): ModuleList(
            (0 - 2): 3 x ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (adain1): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 128, bias = True)
            )
          )
          (adain2): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 128, bias = True)
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
            (0): ParametrizedConv1d(
              32, 32, kernel_size = (3,), stride = (1,), padding = (1,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (1): ParametrizedConv1d(
              32, 32, kernel_size = (3,), stride = (1,), padding = (3,), dilation = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (2): ParametrizedConv1d(
              32, 32, kernel_size = (3,), stride = (1,), padding = (5,), dilation = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (convs2): ModuleList(
            (0 - 2): 3 x ParametrizedConv1d(
              32, 32, kernel_size = (3,), stride = (1,), padding = (1,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (adain1): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 64, bias = True)
            )
          )
          (adain2): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 64, bias = True)
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
            (0): ParametrizedConv1d(
              32, 32, kernel_size = (7,), stride = (1,), padding = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (1): ParametrizedConv1d(
              32, 32, kernel_size = (7,), stride = (1,), padding = (9,), dilation = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (2): ParametrizedConv1d(
              32, 32, kernel_size = (7,), stride = (1,), padding = (15,), dilation = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (convs2): ModuleList(
            (0 - 2): 3 x ParametrizedConv1d(
              32, 32, kernel_size = (7,), stride = (1,), padding = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (adain1): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 64, bias = True)
            )
          )
          (adain2): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 64, bias = True)
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
        (0): Conv1d(22, 64, kernel_size = (np.int64(12),), stride = (np.int64(6),), padding = (np.int64(3),))
        (1): Conv1d(22, 32, kernel_size = (1,), stride = (1,))
      )
      (noise_res): ModuleList(
        (0): AdaINResBlock(
          (convs1): ModuleList(
            (0): ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (1): ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (9,), dilation = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (2): ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (15,), dilation = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (convs2): ModuleList(
            (0 - 2): 3 x ParametrizedConv1d(
              64, 64, kernel_size = (7,), stride = (1,), padding = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (adain1): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 128, bias = True)
            )
          )
          (adain2): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(64, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 128, bias = True)
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
            (0): ParametrizedConv1d(
              32, 32, kernel_size = (11,), stride = (1,), padding = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (1): ParametrizedConv1d(
              32, 32, kernel_size = (11,), stride = (1,), padding = (15,), dilation = (3,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
            (2): ParametrizedConv1d(
              32, 32, kernel_size = (11,), stride = (1,), padding = (25,), dilation = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (convs2): ModuleList(
            (0 - 2): 3 x ParametrizedConv1d(
              32, 32, kernel_size = (11,), stride = (1,), padding = (5,)
                (parametrizations): ModuleDict(
                  (weight): ParametrizationList(
                    (0): _WeightNorm()
                  )
                )
            )
          )
          (adain1): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 64, bias = True)
            )
          )
          (adain2): ModuleList(
            (0 - 2): 3 x AdaIN1d(
              (norm): InstanceNorm1d(32, eps = 1e-05, momentum = 0.1, affine = False, track_running_stats = False)
              (fc): Linear(in_features = 128, out_features = 64, bias = True)
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
      (conv_post): ParametrizedConv1d(
        32, 22, kernel_size = (7,), stride = (1,), padding = (3,)
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
      )
      (reflection_pad): ReflectionPad1d((1, 0))
      (stft): CustomSTFT()
    )
  )
)
```